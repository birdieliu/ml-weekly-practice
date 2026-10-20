import matplotlib.pyplot as plt

from pathlib import Path

import pandas as pd


def save_step_trace(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(records)
    output_path = output_dir / "step_trace.csv"

    table.to_csv(output_path, index=False)

    return output_path

def save_mlp_results(histories, metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存两组各 200 轮的记录
    history_rows = []

    for name, history in histories.items():
        lr = next(
            row["lr"]
            for row in metrics
            if row["model"] == name
        )

        for record in history:
            history_rows.append({
                "config": name,
                "lr": lr,
                "epoch": record["epoch"],
                "train_mse": record["train_mse"],
                "val_mse": record["val_mse"],
                "target_scale": "standardized",
            })

    pd.DataFrame(history_rows).to_csv(
        output_dir / "history.csv",
        index=False,
    )

    pd.DataFrame(metrics).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    # 双面板训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, name in zip(axes, ("SGD", "Adam")):
        history = histories[name]

        row = next(
            row for row in metrics
            if row["model"] == name
        )

        epochs = [record["epoch"] for record in history]

        ax.plot(
            epochs,
            [record["train_mse"] for record in history],
            label="Train",
        )

        ax.plot(
            epochs,
            [record["val_mse"] for record in history],
            label="Validation",
        )

        ax.axvline(
            row["best_epoch"],
            color="black",
            linestyle="--",
            label=f"Best epoch: {row['best_epoch']}",
        )

        ax.set_title(f"{name}, lr={row['lr']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (standardized target)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # 四组模型的训练、验证 RMSE
    positions = list(range(len(metrics)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [position - width / 2 for position in positions],
        [row["train_rmse"] for row in metrics],
        width=width,
        label="Train",
    )

    ax.bar(
        [position + width / 2 for position in positions],
        [row["val_rmse"] for row in metrics],
        width=width,
        label="Validation",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([row["model"] for row in metrics])
    ax.set_ylabel("RMSE (original target units)")
    ax.set_title("MLPs evaluated at their best validation epochs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "rmse_comparison.png", dpi=150)
    plt.close(fig)

    print("\n正式结果已保存到：", output_dir)