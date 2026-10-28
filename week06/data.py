import numpy as np


def make_windows(series, window_size):
    """用连续 window_size 个历史值，预测紧接着的下一个值。"""
    series = np.asarray(series, dtype=float)

    if series.ndim != 1:
        raise ValueError("series 必须是一维序列")
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size 必须是正整数")
    if window_size >= len(series):
        raise ValueError("窗口长度必须小于序列长度")

    X = []
    y = []
    target_index = []

    for t in range(window_size, len(series)):
        X.append(series[t - window_size:t])
        y.append(series[t])
        target_index.append(t)

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(target_index),
    )
def load_series(csv_path, n_hours=14400):
    """读取实验时段，并检查时间和数值。"""
    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=["date"])

    if len(df) < n_hours:
        raise ValueError("数据行数不足")

    df = df.iloc[:n_hours].copy()

    if df["date"].isna().any():
        raise ValueError("存在无效时间")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("时间没有按先后顺序排列")

    if df["date"].duplicated().any():
        raise ValueError("存在重复时间")

    intervals = df["date"].diff().iloc[1:]
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("相邻记录不是间隔1小时")

    numeric_values = df.drop(columns="date").to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("数值中存在缺失值或无穷值")

    series = df["OT"].to_numpy(dtype=float)
    times = df["date"].to_numpy()

    return df, series, times
def split_by_target_time(times, target_index, boundaries):
    """按目标时间生成训练、验证、测试掩码。"""
    import pandas as pd

    target_times = pd.DatetimeIndex(times[target_index])
    masks = {}

    for split, (start, stop) in boundaries.items():
        masks[split] = (
            (target_times >= pd.Timestamp(start))
            & (target_times < pd.Timestamp(stop))
        )

    # 每条样本必须恰好归入一个集合
    membership_count = np.zeros(len(target_index), dtype=int)

    for mask in masks.values():
        membership_count += mask.astype(int)

    if not np.all(membership_count == 1):
        raise ValueError("划分存在重复归组或遗漏样本")

    return masks