from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 让 Python 能从仓库根目录找到 week06
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week06.data import (
    load_series,
    make_windows,
    split_by_target_time,
)


def prepare_data():
    data_path = ROOT / "week06" / "data" / "ETTh1.csv"
    df, series, times = load_series(data_path, n_hours=14400)

    # 先构造窗口，再按目标时间划分
    X_raw, y_raw, target_index = make_windows(
        series, window_size=24
    )

    boundaries = {
        "train": ("2016-07-01", "2017-06-26"),
        "val": ("2017-06-26", "2017-10-24"),
        "test": ("2017-10-24", "2018-02-21"),
    }

    masks = split_by_target_time(
        times, target_index, boundaries
    )

    # 从原始训练时段计算统计量，每个时刻只计一次
    train_times = (
        (df["date"] >= pd.Timestamp(boundaries["train"][0]))
        & (df["date"] < pd.Timestamp(boundaries["train"][1]))
    )
    train_values = series[train_times.to_numpy()]

    mu = float(train_values.mean())
    sigma = float(train_values.std(ddof=0))

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("训练标准差必须为有限正数")

    # 输入：(N, 24) → (N, 24, 1)
    X = ((X_raw - mu) / sigma).astype(np.float32)
    X = X[:, :, None]

    # 目标：(N,) → (N, 1)
    y = ((y_raw - mu) / sigma).astype(np.float32)
    y = y[:, None]

    splits = {}
    for name, mask in masks.items():
        splits[name] = {
            "X": X[mask],
            "y": y[mask],
            "y_raw": y_raw[mask],
            "target_index": target_index[mask],
            "target_time": times[target_index[mask]],
        }

    return splits, mu, sigma


def main():
    splits, mu, sigma = prepare_data()

    print(f"训练均值 mu：{mu:.8f}")
    print(f"训练总体标准差 sigma：{sigma:.8f}")

    for name, part in splits.items():
        print(
            f"\n{name}: "
            f"X={part['X'].shape}, "
            f"y={part['y'].shape}, "
            f"dtype={part['X'].dtype}"
        )
        print(
            "目标时间范围：",
            part["target_time"][0],
            "至",
            part["target_time"][-1],
        )

    # 独立从原始数据核对第一条验证样本
    df, series, times = load_series(
        ROOT / "week06" / "data" / "ETTh1.csv"
    )
    val = splits["val"]
    t = int(val["target_index"][0])

    restored_window = val["X"][0, :, 0] * sigma + mu
    restored_target = val["y"][0, 0] * sigma + mu

    np.testing.assert_allclose(
        restored_window, series[t - 24:t],
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(
        restored_target, series[t],
        rtol=1e-5, atol=1e-5,
    )

    print("\n第一条验证样本核对：")
    print("目标下标：", t)
    print("输入开始：", times[t - 24])
    print("输入结束：", times[t - 1])
    print("输入首尾原始油温：", series[t - 24], series[t - 1])
    print("目标时间：", times[t])
    print("目标原始油温：", series[t])
    print("标准化目标：", val["y"][0, 0])
    print("还原后窗口与目标核对通过")


if __name__ == "__main__":
    main()