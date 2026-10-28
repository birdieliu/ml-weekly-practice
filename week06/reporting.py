from pathlib import Path

import pandas as pd


def save_window_examples(records, output_path):
    """把窗口记录保存为 CSV，并返回保存的表格。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(records)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")

    return table
def save_experiment_tables(tuning, metrics, predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "tuning.csv": tuning,
        "metrics.csv": metrics,
        "predictions.csv": predictions,
    }

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )
def plot_experiment(df, tuning, predictions, window_size, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    output_dir = Path(output_dir)

    # macOS 中文字体
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：不同配置的训练、验证误差
    labels = [
        "lag_linear" if row.model == "lag_linear"
        else f"Ridge alpha={float(row.param_value):g}"
        for row in tuning.itertuples(index=False)
    ]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(positions, tuning["train_rmse"], "o-", label="训练")
    ax.plot(positions, tuning["val_rmse"], "o-", label="验证")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE（原始油温尺度）")
    ax.set_xlabel("模型配置")
    ax.set_title("普通线性回归与 Ridge 超参数对照")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tuning_comparison.png", dpi=180)
    plt.close(fig)

    # 图2：整个实验时段与划分边界
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["OT"], linewidth=0.6, label="油温 OT")

    ax.axvline(
        pd.Timestamp("2017-06-26"),
        color="orange", linestyle="--", label="验证起点",
    )
    ax.axvline(
        pd.Timestamp("2017-10-24"),
        color="red", linestyle="--", label="测试起点",
    )

    # 初始24条历史：从首个输入到首个目标之前
    ax.axvspan(
        df["date"].iloc[0],
        df["date"].iloc[window_size],
        color="green", alpha=0.3,
        label=f"初始历史（{window_size}个小时观测）",
    )
    ax.annotate(
        "初始历史",
        xy=(df["date"].iloc[window_size // 2], df["OT"].iloc[0]),
        xytext=(45, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_ylabel("油温 OT（原始数据尺度）")
    ax.set_xlabel("时间")
    ax.set_title("实验时段与数据划分")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "series_split.png", dpi=180)
    plt.close(fig)

    # 图3：验证首周；油温来自预测表，负载来自原始表
    start = pd.Timestamp("2017-06-26")
    stop = pd.Timestamp("2017-07-03")

    week = predictions[
        (predictions["split"] == "val")
        & (predictions["target_time"] >= start)
        & (predictions["target_time"] < stop)
    ].copy()

    model_order = [
        "last_value", "window_mean", "lag_linear", "ridge_tuned"
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    truth = week[week["model"] == "last_value"].sort_values("target_time")
    assert len(truth) == 168
    axes[0].plot(
        truth["target_time"], truth["y_true"],
        color="black", linewidth=1.8, label="真实油温",
    )

    for name in model_order:
        part = week[week["model"] == name].sort_values("target_time")
        assert len(part) == 168
        axes[0].plot(
            part["target_time"], part["y_pred"],
            linewidth=1, alpha=0.8, label=name,
        )

    # 按同一组真实时间取负载，防止错位
    load = df.set_index("date").loc[truth["target_time"], "HUFL"]
    axes[1].plot(
        truth["target_time"], load.to_numpy(),
        color="tab:purple", label="HUFL",
    )

    axes[0].set_ylabel("油温 OT（原始数据尺度）")
    axes[1].set_ylabel("HUFL（原始数据尺度）")
    axes[1].set_xlabel("时间")
    axes[0].set_title("验证首周：单步油温预测与负载观察")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d\n%H:%M")
    )
    fig.tight_layout()
    fig.savefig(output_dir / "validation_predictions.png", dpi=180)
    plt.close(fig)