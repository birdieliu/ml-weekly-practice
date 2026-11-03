from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sequence_data import prepare_data


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "week07" / "results"

    splits, _, _ = prepare_data()

    checkpoint = torch.load(
        output_dir / "chosen_lstm.pt",
        map_location="cpu",
        weights_only=True,
    )
    chosen_name = f"lstm_h{checkpoint['hidden_size']}"

    # 即使重复运行，也只取其中的 LSTM 记录
    current = pd.read_csv(
        output_dir / "predictions.csv",
        parse_dates=["target_time"],
    )
    lstm = current[
        current["model"].isin(["lstm_h16", "lstm_h32"])
    ].copy()

    # 复用上周两种基线，保留其原有预处理结果
    previous = pd.read_csv(
        root / "week06" / "results" / "predictions.csv",
        parse_dates=["target_time"],
        encoding="utf-8-sig",
    )
    baselines = previous[
        previous["model"].isin(["last_value", "lag_linear"])
    ].copy()

    predictions = pd.concat(
        [baselines, lstm], ignore_index=True
    )

    keys = ["model", "split", "target_index"]
    if predictions.duplicated(keys).any():
        raise ValueError("存在重复预测")

    # 明确规定每个集合应包含哪些模型
    expected_models = {
        "train": {"last_value", "lag_linear", "lstm_h16", "lstm_h32"},
        "val": {"last_value", "lag_linear", "lstm_h16", "lstm_h32"},
        "test": {"last_value", "lag_linear", chosen_name},
    }

    if set(predictions["split"]) != set(expected_models):
        raise ValueError("数据部分不符合要求")

    history = pd.read_csv(output_dir / "history.csv")
    best_epochs = {}

    for name in ("lstm_h16", "lstm_h32"):
        rows = history[history["model"] == name].sort_values("epoch")

        if rows["epoch"].tolist() != list(range(1, 41)):
            raise ValueError(f"{name} 的训练轮次记录不完整")

        # idxmin 在完全同分时返回排在前面的轮次
        best_epochs[name] = int(
            rows.loc[rows["val_mse"].idxmin(), "epoch"]
        )

    metrics = []

    for split, model_names in expected_models.items():
        split_rows = predictions[predictions["split"] == split]

        if set(split_rows["model"]) != model_names:
            raise ValueError(f"{split} 的模型集合不符合要求")

        part = splits[split]

        for name in sorted(model_names):
            rows = split_rows[
                split_rows["model"] == name
            ].sort_values("target_index")

            # 核对全部目标：无遗漏，且与本周数据对应
            np.testing.assert_array_equal(
                rows["target_index"].to_numpy(),
                part["target_index"],
            )
            np.testing.assert_array_equal(
                rows["target_time"].to_numpy(),
                part["target_time"],
            )
            np.testing.assert_allclose(
                rows["y_true"].to_numpy(),
                part["y_raw"],
                rtol=0,
                atol=1e-10,
            )

            values = rows[["y_true", "y_pred"]].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("预测或真实值存在无效数值")

            error = values[:, 1] - values[:, 0]
            is_lstm = name.startswith("lstm_h")

            metrics.append({
                "model": name,
                "split": split,
                "n_samples": len(rows),
                "hidden_size": (
                    int(name.removeprefix("lstm_h"))
                    if is_lstm else None
                ),
                "best_epoch": (
                    best_epochs[name] if is_lstm else None
                ),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "chosen_lstm": name == chosen_name,
            })

    predictions = predictions.sort_values(
        ["model", "split", "target_index"]
    )
    metrics = pd.DataFrame(metrics)

    predictions.to_csv(
        output_dir / "predictions.csv", index=False
    )
    metrics.to_csv(
        output_dir / "metrics.csv", index=False
    )

    print("目标下标、时间、真实值与预测完整性核对通过。")
    print(f"预测总行数：{len(predictions)}")
    print("\n原始油温尺度指标：")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()