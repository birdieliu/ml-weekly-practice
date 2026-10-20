from pathlib import Path

from reporting import save_mlp_results


import copy

import torch

from data import prepare_tensor_data, create_train_loader
from models import create_mlp
from optimizers import create_optimizer
from training import fit_baselines, train_mlp
from evaluation import (
    evaluate_mean_baseline,
    evaluate_linear_baseline,
    evaluate_mlp_raw,
)

def check_hidden_width(data):
    temporary_model = create_mlp(hidden_dim=8)

    # 只取训练集中的 32 条输入，不训练
    xb = data["X_train_tensor"][:32]

    print("\n隐藏层宽度 8 的前向检查：")
    print("输入形状：", xb.shape)

    with torch.no_grad():
        output = xb

        for layer in temporary_model:
            output = layer(output)

            print(
                "经过",
                layer.__class__.__name__,
                "后的形状：",
                output.shape,
            )

    assert output.shape == (len(xb), 1)

    print("检查通过：隐藏层宽度改变后，每条样本仍输出一个预测")


def main():
    # 所有模型共用这一份数据划分和标准化结果
    data = prepare_tensor_data()

    # 拟合两个基线
    mean_value, linear_model = fit_baselines(data)

    metrics = []

    for name in ("Mean", "LinearRegression"):
        row = {
            "model": name,
            "lr": None,
            "best_epoch": None,
        }

        for split in ("train", "val"):
            X = data[f"X_{split}"]
            y_raw = data[f"y_{split}_raw"]

            if name == "Mean":
                scores = evaluate_mean_baseline(
                    mean_value, X, y_raw
                )
            else:
                scores = evaluate_linear_baseline(
                    linear_model, X, y_raw
                )

            row[f"{split}_mae"] = scores["mae"]
            row[f"{split}_rmse"] = scores["rmse"]

        metrics.append(row)

    # 创建一个未训练的初始网络
    torch.manual_seed(42)
    initial_model = create_mlp(hidden_dim=16)

    # 从同一个初始网络复制出两个独立模型
    models = {
        "SGD": copy.deepcopy(initial_model),
        "Adam": copy.deepcopy(initial_model),
    }

    for sgd_param, adam_param in zip(
        models["SGD"].parameters(),
        models["Adam"].parameters(),
    ):
        assert torch.equal(sgd_param, adam_param)

    print("检查通过：SGD 和 Adam 的初始参数相同")

    histories = {}

    for name, lr in (("SGD", 0.01), ("Adam", 0.001)):
        print(f"\n开始训练 {name}，学习率为 {lr}")

        model = models[name]

        # 每组分别创建 DataLoader，使每轮打乱顺序一致
        loader = create_train_loader(data, batch_size=32)

        optimizer = create_optimizer(
            model,
            name=name,
            lr=lr,
        )

        history, best_epoch, best_val_mse = train_mlp(
            model,
            optimizer,
            loader,
            data,
            epochs=200,
        )

        histories[name] = history

        row = {
            "model": name,
            "lr": lr,
            "best_epoch": best_epoch,
        }

        for split in ("train", "val"):
            scores = evaluate_mlp_raw(
                model,
                data[f"X_{split}_tensor"],
                data[f"y_{split}_raw"],
                data["y_mean"],
                data["y_std"],
            )

            row[f"{split}_mae"] = scores["mae"]
            row[f"{split}_rmse"] = scores["rmse"]

        metrics.append(row)

        print("最佳轮次：", best_epoch)
        print("最佳验证 MSE：", best_val_mse)

    # 只在两个 MLP 之间选择；相同则选 SGD
    mlp_rows = {
        row["model"]: row
        for row in metrics
        if row["model"] in ("SGD", "Adam")
    }

    if (
        mlp_rows["SGD"]["val_rmse"]
        <= mlp_rows["Adam"]["val_rmse"]
    ):
        selected_name = "SGD"
    else:
        selected_name = "Adam"

    print("\n四组模型的原始单位指标：")

    for row in metrics:
        print(
            f"{row['model']}："
            f"训练 RMSE={row['train_rmse']:.4f}，"
            f"验证 RMSE={row['val_rmse']:.4f}"
        )

    print("\n根据验证 RMSE 选定的 MLP：", selected_name)
        # 所有模型先建立测试列；未评价的保持为空
    for row in metrics:
        row["test_mae"] = None
        row["test_rmse"] = None
        row["selected_mlp"] = row["model"] == selected_name
        row["metric_scale"] = "original"

    # 只评价两个基线和选定的 MLP
    for row in metrics:
        name = row["model"]

        if name == "Mean":
            scores = evaluate_mean_baseline(
                mean_value,
                data["X_test"],
                data["y_test_raw"],
            )

        elif name == "LinearRegression":
            scores = evaluate_linear_baseline(
                linear_model,
                data["X_test"],
                data["y_test_raw"],
            )

        elif name == selected_name:
            scores = evaluate_mlp_raw(
                models[name],
                data["X_test_tensor"],
                data["y_test_raw"],
                data["y_mean"],
                data["y_std"],
            )

        else:
            continue

        row["test_mae"] = scores["mae"]
        row["test_rmse"] = scores["rmse"]

    print("\n最终测试指标，原始单位：")

    for row in metrics:
        if row["test_rmse"] is not None:
            print(
                f"{row['model']}："
                f"MAE={row['test_mae']:.4f}，"
                f"RMSE={row['test_rmse']:.4f}"
            )

    output_dir = Path(__file__).resolve().parent / "results"

    save_mlp_results(
        histories,
        metrics,
        output_dir,
    )
    check_hidden_width(data)


if __name__ == "__main__":
    main()