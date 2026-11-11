from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from week07.sequence_data import prepare_data
from week07.training import make_loader, evaluate_mse
from week08.models import SingleHeadForecaster


def main():
    torch.manual_seed(42)

    # 复用上周的数据划分和训练时段标准化统计量
    splits, mu, sigma = prepare_data()

    train_loader = make_loader(splits["train"], shuffle=True)
    train_eval_loader = make_loader(splits["train"], shuffle=False)
    val_loader = make_loader(splits["val"], shuffle=False)

    model = SingleHeadForecaster(d_model=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, 41):
        model.train()

        for x, y in train_loader:
            optimizer.zero_grad()

            prediction = model(x)
            loss = loss_function(prediction, y)

            loss.backward()
            optimizer.step()

        # 用本轮结束后的参数评价完整训练集、验证集
        train_mse = evaluate_mse(model, train_eval_loader)
        val_mse = evaluate_mse(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })

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

    # 恢复最佳轮次，而不是直接保存第40轮
    model.load_state_dict(best_state)
    model.eval()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "d_model": 16,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "mu": mu,
            "sigma": sigma,
            "window_size": 24,
        },
        output_dir / "best_attention.pt",
    )

    pd.DataFrame(history).to_csv(
        output_dir / "history.csv", index=False
    )

    # 还原到原始油温尺度，再计算验证指标
    predictions = []
    with torch.no_grad():
        for x, _ in val_loader:
            predictions.append(model(x))

    prediction_raw = (
        torch.cat(predictions).numpy().reshape(-1) * sigma + mu
    )
    target_raw = splits["val"]["y_raw"].reshape(-1)
    error = prediction_raw - target_raw

    mae = float(abs(error).mean())
    rmse = float((error ** 2).mean() ** 0.5)

    pd.DataFrame([{
        "model": "single_head_attention",
        "val_samples": len(target_raw),
        "best_epoch": best_epoch,
        "mae": mae,
        "rmse": rmse,
    }]).to_csv(output_dir / "metrics.csv", index=False)

    print(f"\n最佳轮次：{best_epoch}")
    print(f"训练均值：{mu:.8f}")
    print(f"训练标准差：{sigma:.8f}")
    print(f"验证样本数：{len(target_raw)}")
    print(f"验证 MAE（原始尺度）：{mae:.6f}")
    print(f"验证 RMSE（原始尺度）：{rmse:.6f}")
    print(f"结果保存至：{output_dir}")


if __name__ == "__main__":
    main()