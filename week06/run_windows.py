from pathlib import Path

import numpy as np
import pandas as pd

from data import make_windows
from reporting import save_window_examples


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "ETTh1.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])

    series = df["OT"].iloc[:12].to_numpy(dtype=float)
    times = df["date"].iloc[:12].to_numpy()

    records = []

    for window_size in [3, 5]:
        X, y, target_index = make_windows(series, window_size)

        # 核对样本数量、输入内容和目标
        assert X.shape == (len(series) - window_size, window_size)
        assert y.shape == target_index.shape == (len(series) - window_size,)

        for row, t in enumerate(target_index):
            start = t - window_size
            end = t - 1

            np.testing.assert_array_equal(X[row], series[start:t])
            assert y[row] == series[t]

            # 输入的每个时刻都必须早于目标时刻
            assert np.all(times[start:t] < times[t])

            records.append({
                "window_size": window_size,
                "input_start": int(start),
                "input_end": int(end),
                "target_index": int(t),
                "target_time": pd.Timestamp(times[t]),
                "input_values": X[row].tolist(),
                "target_value": float(y[row]),
            })

        print(f"\n窗口长度：{window_size}")
        print("X 的形状：", X.shape)
        print("y 的形状：", y.shape)
        print("target_index 的形状：", target_index.shape)

        for row in [0, len(y) - 1]:
            t = target_index[row]
            start = t - window_size
            end = t - 1

            print(f"\n第 {row + 1} 条样本")
            print("输入开始时间：", pd.Timestamp(times[start]))
            print("输入结束时间：", pd.Timestamp(times[end]))
            print("输入数值：", X[row])
            print("目标下标：", t)
            print("目标时间：", pd.Timestamp(times[t]))
            print("目标数值：", y[row])

    output_path = base_dir / "results" / "window_examples.csv"
    table = save_window_examples(records, output_path)

    print("\n各窗口长度的记录数：")
    print(table.groupby("window_size").size().to_string())
    print("\n总记录数：", len(table))
    print("已保存到：", output_path)


if __name__ == "__main__":
    main()