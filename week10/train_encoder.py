from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from week07.sequence_data import prepare_data
from week07.training import make_loader, evaluate_mse
from week10.encoder import EncoderForecaster


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    splits, mu, sigma = prepare_data()

    train_loader = make_loader(splits["train"], shuffle=True)
    train_eval_loader = make_loader(splits["train"], shuffle=False)
    val_loader = make_loader(splits["val"], shuffle=False)

    model = EncoderForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    history = []
    best_val_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, 41):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()

        train_mse = evaluate_mse(model, train_eval_loader)
        val_mse = evaluate_mse(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })

        # 严格小于才更新，因此同分时保留更早轮次
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        print(
            f"epoch={epoch:02d}/40, "
            f"train_mse={train_mse:.6f}, "
            f"val_mse={val_mse:.6f}",
            flush=True,
        )

    # 恢复验证 MSE 最佳轮次参数
    model.load_state_dict(best_state)
    model.eval()

    checkpoint_path = results_dir / "best_encoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "d_model": 16,
            "d_head": 8,
            "d_ff": 32,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "mu": mu,
            "sigma": sigma,
            "window_size": 24,
            "seed": 42,
        },
        checkpoint_path,
    )

    pd.DataFrame(history).to_csv(
        results_dir / "history.csv", index=False
    )

    # 恢复到原始油温尺度，在同一批验证目标上计算 MAE/RMSE
    predictions = []
    with torch.no_grad():
        for x, _ in val_loader:
            prediction = model(x.to(device))
            predictions.append(prediction.cpu())

    prediction_standardized = torch.cat(predictions).numpy().reshape(-1)
    prediction_raw = prediction_standardized * sigma + mu
    target_raw = splits["val"]["y_raw"].reshape(-1)

    error = prediction_raw - target_raw
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.mean(error ** 2)))

    metrics = pd.DataFrame([{
        "model": "week10_single_encoder_layer",
        "val_samples": len(target_raw),
        "best_epoch": best_epoch,
        "best_val_mse_standardized": best_val_mse,
        "mae_raw": mae,
        "rmse_raw": rmse,
    }])
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    print(f"\n最佳轮次：{best_epoch}")
    print(f"最佳标准化验证 MSE：{best_val_mse:.6f}")
    print(f"验证样本数：{len(target_raw)}")
    print(f"验证 MAE（原始油温尺度）：{mae:.6f}")
    print(f"验证 RMSE（原始油温尺度）：{rmse:.6f}")
    print(f"最佳模型保存至：{checkpoint_path}")


if __name__ == "__main__":
    main()