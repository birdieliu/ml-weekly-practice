from evaluation import make_prediction_table

from pathlib import Path

import pandas as pd
import torch

from sequence_data import prepare_data
from training import fit_one_model


def main():
    splits, mu, sigma = prepare_data()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_history = []
    prediction_tables = []
    summaries = []
    chosen = None

    for hidden_size in (16, 32):
        print(f"\n开始训练 hidden_size={hidden_size}", flush=True)

        model, history, best_epoch, best_val_mse = (
            fit_one_model(splits, hidden_size)
        )

        all_history.extend(history)
                # fit_one_model 已恢复该组的最佳参数
        # 两种宽度都保存训练、验证预测
        for split in ("train", "val"):
            table = make_prediction_table(
                model=model,
                part=splits[split],
                mu=mu,
                sigma=sigma,
                model_name=f"lstm_h{hidden_size}",
                split=split,
            )
            prediction_tables.append(table)

        # 标准化尺度 MSE → 原始油温尺度 RMSE
        val_rmse = (best_val_mse ** 0.5) * sigma

        summaries.append({
            "hidden_size": hidden_size,
            "best_epoch": best_epoch,
            "val_mse": best_val_mse,
            "val_rmse": val_rmse,
        })

        # 先运行16；严格更好才替换，完全同分时保留16
        if chosen is None or val_rmse < chosen["val_rmse"]:
            chosen = {
                "model": model,
                "hidden_size": hidden_size,
                "best_epoch": best_epoch,
                "val_rmse": val_rmse,
            }

        # 每完成一组就保存已有训练记录
        pd.DataFrame(all_history).to_csv(
            output_dir / "history.csv",
            index=False,
        )
        # 选择已固定，只评价选定 LSTM 的测试集
    test_table = make_prediction_table(
        model=chosen["model"],
        part=splits["test"],
        mu=mu,
        sigma=sigma,
        model_name=f"lstm_h{chosen['hidden_size']}",
        split="test",
    )
    prediction_tables.append(test_table)

    predictions = pd.concat(prediction_tables, ignore_index=True)

    if predictions.duplicated(
        ["model", "split", "target_index"]
    ).any():
        raise ValueError("存在重复预测")

    expected_rows = (
        2 * (len(splits["train"]["y"]) + len(splits["val"]["y"]))
        + len(splits["test"]["y"])
    )
    if len(predictions) != expected_rows:
        raise ValueError("预测记录数量不正确")

    predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    print(f"\nLSTM 预测已保存，共 {len(predictions)} 行。")
    
    # 只保存选定模型，不保存每轮模型
    checkpoint_path = output_dir / "chosen_lstm.pt"
    torch.save(
        {
            "model_state_dict": chosen["model"].state_dict(),
            "hidden_size": chosen["hidden_size"],
            "best_epoch": chosen["best_epoch"],
            "mu": mu,
            "sigma": sigma,
            "window_size": 24,
            "input_size": 1,
            "num_layers": 1,
        },
        checkpoint_path,
    )

    print("\n两组最佳验证结果（RMSE为原始油温尺度）：")
    print(pd.DataFrame(summaries).to_string(index=False))

    print(
        f"\n选定 LSTM：hidden_size={chosen['hidden_size']}, "
        f"best_epoch={chosen['best_epoch']}, "
        f"val_rmse={chosen['val_rmse']:.6f}"
    )
    print("history.csv 中的 MSE 为标准化尺度。")
    print(f"选定模型已保存：{checkpoint_path}")


if __name__ == "__main__":
    main()