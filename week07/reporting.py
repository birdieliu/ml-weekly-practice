from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves(history, metrics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, hidden_size in zip(axes, (16, 32)):
        name = f"lstm_h{hidden_size}"
        rows = history[
            history["model"] == name
        ].sort_values("epoch")

        best_epoch = int(
            metrics.loc[
                (metrics["model"] == name)
                & (metrics["split"] == "val"),
                "best_epoch",
            ].iloc[0]
        )

        best_val = rows.loc[
            rows["epoch"] == best_epoch, "val_mse"
        ].iloc[0]

        ax.plot(rows["epoch"], rows["train_mse"], label="Train")
        ax.plot(rows["epoch"], rows["val_mse"], label="Validation")

        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best epoch: {best_epoch}",
        )
        ax.scatter(
            best_epoch, best_val, color="red", zorder=3
        )

        ax.set_title(f"LSTM hidden size = {hidden_size}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (standardized scale)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_validation_predictions(predictions, metrics, output_dir):
    chosen_names = metrics.loc[
        metrics["chosen_lstm"], "model"
    ].unique()

    if len(chosen_names) != 1:
        raise ValueError("应当恰好选定一种 LSTM")

    chosen_name = chosen_names[0]
    models = ["last_value", "lag_linear", chosen_name]

    start = pd.Timestamp("2017-06-26")
    stop = pd.Timestamp("2017-07-03")

    week = predictions[
        (predictions["split"] == "val")
        & (predictions["target_time"] >= start)
        & (predictions["target_time"] < stop)
    ]

    reference = week[
        week["model"] == "last_value"
    ].sort_values("target_time")

    expected_times = pd.date_range(
        start=start, periods=168, freq="h"
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        reference["target_time"],
        reference["y_true"],
        color="black",
        linewidth=2,
        label="True OT",
    )

    for name in models:
        rows = week[
            week["model"] == name
        ].sort_values("target_time")

        if not pd.DatetimeIndex(
            rows["target_time"]
        ).equals(expected_times):
            raise ValueError(f"{name} 的首周时间记录不完整")

        ax.plot(
            rows["target_time"],
            rows["y_pred"],
            label=name,
            alpha=0.8,
        )

    ax.set_title("Validation predictions: first week")
    ax.set_xlabel("Target time")
    ax.set_ylabel("OT (original scale)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        output_dir / "validation_predictions.png", dpi=180
    )
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent / "results"

    history = pd.read_csv(output_dir / "history.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    predictions = pd.read_csv(
        output_dir / "predictions.csv",
        parse_dates=["target_time"],
    )

    plot_loss_curves(history, metrics, output_dir)
    plot_validation_predictions(predictions, metrics, output_dir)

    print("已生成：", output_dir / "loss_curves.png")
    print("已生成：", output_dir / "validation_predictions.png")


if __name__ == "__main__":
    main()