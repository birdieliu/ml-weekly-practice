from reporting import plot_experiment

import numpy as np

from reporting import save_experiment_tables

from pathlib import Path

import pandas as pd

from data import load_series, make_windows, split_by_target_time
from models import (
    predict_last_value,
    predict_window_mean,
    create_lag_linear,
)
from evaluation import calculate_metrics
from models import create_ridge

def main():
    base_dir = Path(__file__).resolve().parent

    # 正式实验的固定设置
    n_hours = 14400
    window_size = 24

    df, series, times = load_series(
        base_dir / "data" / "ETTh1.csv",
        n_hours=n_hours,
    )

    X, y, target_index = make_windows(series, window_size)

    print("数据检查通过")
    print("使用记录数：", len(df))
    print("数据开始时间：", pd.Timestamp(times[0]))
    print("数据结束时间：", pd.Timestamp(times[-1]))
    print("X 的形状：", X.shape)
    print("y 的形状：", y.shape)

    # 核对正式实验的第一条样本
    t = target_index[0]
    start = t - window_size
    end = t - 1

    print("\n第一条样本的完整历史：")
    print(df.iloc[start:t][["date", "OT"]].to_string(index=False))

    print("\n输入开始时间：", pd.Timestamp(times[start]))
    print("输入结束时间：", pd.Timestamp(times[end]))
    print("目标下标：", t)
    print("目标时间：", pd.Timestamp(times[t]))
    print("目标油温：", y[0])
        # 按任务规定的目标时间划分
    boundaries = {
        "train": ("2016-07-01", "2017-06-26"),
        "val": ("2017-06-26", "2017-10-24"),
        "test": ("2017-10-24", "2018-02-21"),
    }

    masks = split_by_target_time(times, target_index, boundaries)

    print("\n各集合的样本数：")
    for split, mask in masks.items():
        print(f"{split}: {int(mask.sum())}")

    # 核对验证集、测试集的第一条样本
    for split in ["val", "test"]:
        first_target = target_index[masks[split]][0]
        start = first_target - window_size
        end = first_target - 1

        print(f"\n{split} 第一条样本")
        print("输入开始时间：", pd.Timestamp(times[start]))
        print("输入结束时间：", pd.Timestamp(times[end]))
        print("目标下标：", first_target)
        print("目标时间：", pd.Timestamp(times[first_target]))
        # 线性模型只在训练集上拟合
    linear_model = create_lag_linear()
    linear_model.fit(X[masks["train"]], y[masks["train"]])

    # 三种方法统一接收原始油温窗口
    predictors = {
        "last_value": predict_last_value,
        "window_mean": predict_window_mean,
        "lag_linear": linear_model.predict,
    }

    metric_records = []

    for model_name, predict in predictors.items():
        for split in ["train", "val"]:
            mask = masks[split]
            y_pred = predict(X[mask])
            scores = calculate_metrics(y[mask], y_pred)

            metric_records.append({
                "model": model_name,
                "split": split,
                "n_samples": int(mask.sum()),
                **scores,
            })

    print("\n三种基线的训练与验证指标：")
    metric_table = pd.DataFrame(metric_records)
    print(metric_table.to_string(index=False, float_format="%.6f"))

    # 用第一条验证样本，核对一次具体预测
    example_X = X[masks["val"]][:1]
    example_y = y[masks["val"]][0]
    example_t = target_index[masks["val"]][0]

    print("\n第一条验证样本的预测对照：")
    print("目标时间：", pd.Timestamp(times[example_t]))
    print("24小时输入：", example_X[0])
    print("真实油温：", example_y)

    for model_name, predict in predictors.items():
        prediction = float(predict(example_X)[0])
        print(
            f"{model_name}: "
            f"预测={prediction:.6f}, "
            f"预测减真实值={prediction - example_y:.6f}"
        )
        # 三个候选只比较训练、验证结果，不查看测试指标
    ridge_alphas = [0.1, 10.0, 1000.0]
    tuning_records = []
    ridge_models = {}

    def tuning_row(model_name, model, param_value):
        train_scores = calculate_metrics(
            y[masks["train"]],
            model.predict(X[masks["train"]]),
        )
        val_scores = calculate_metrics(
            y[masks["val"]],
            model.predict(X[masks["val"]]),
        )

        return {
            "model": model_name,
            "param_name": "alpha" if model_name == "ridge" else "",
            "param_value": param_value,
            "train_mae": train_scores["mae"],
            "val_mae": val_scores["mae"],
            "train_rmse": train_scores["rmse"],
            "val_rmse": val_scores["rmse"],
            "rmse_gap": val_scores["rmse"] - train_scores["rmse"],
            "chosen_candidate": False,
        }

    # 普通线性回归作为参照
    tuning_records.append(tuning_row("lag_linear", linear_model, ""))

    for alpha in ridge_alphas:
        model = create_ridge(alpha)
        model.fit(X[masks["train"]], y[masks["train"]])
        ridge_models[alpha] = model
        tuning_records.append(tuning_row("ridge", model, alpha))

    # 验证 RMSE 最小者胜出；完全相同时取较小 alpha
    best_row = min(
        tuning_records[1:],
        key=lambda row: (row["val_rmse"], row["param_value"]),
    )
    best_row["chosen_candidate"] = True

    best_alpha = best_row["param_value"]
    ridge_tuned = ridge_models[best_alpha]
    tuning_table = pd.DataFrame(tuning_records)

    print("\nRidge 调参结果：")
    print(tuning_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n选定的 alpha：", best_alpha)
        # 固定四种最终方法；字典顺序也是完全同分时的选择顺序
    final_predictors = {
        "last_value": predict_last_value,
        "window_mean": predict_window_mean,
        "lag_linear": linear_model.predict,
        "ridge_tuned": ridge_tuned.predict,
    }

    # 只根据完整验证集选择总体最优方法
    val_rmse = {}
    for name, predict in final_predictors.items():
        scores = calculate_metrics(
            y[masks["val"]],
            predict(X[masks["val"]]),
        )
        val_rmse[name] = scores["rmse"]

    selected_model = min(val_rmse, key=val_rmse.get)
    print("\n验证集选定的最终方法：", selected_model)

    final_metric_records = []
    prediction_parts = []

    # 选择固定后，才计算测试结果；不重新拟合模型
    for name, predict in final_predictors.items():
        for split, mask in masks.items():
            targets = target_index[mask]
            y_pred = predict(X[mask])
            scores = calculate_metrics(y[mask], y_pred)

            final_metric_records.append({
                "model": name,
                "split": split,
                "n_samples": int(mask.sum()),
                "mae": scores["mae"],
                "rmse": scores["rmse"],
                "selected": name == selected_model,
            })

            prediction_parts.append(pd.DataFrame({
                "model": name,
                "split": split,
                "target_index": targets,
                "target_time": times[targets],
                "y_true": y[mask],
                "y_pred": y_pred,
            }))

    metrics_table = pd.DataFrame(final_metric_records)
    predictions_table = pd.concat(prediction_parts, ignore_index=True)

    # 每种方法必须覆盖全部目标，且每个目标只出现一次
    assert len(predictions_table) == 4 * len(y)
    assert not predictions_table.duplicated(
        ["model", "target_index"]
    ).any()

    for name in final_predictors:
        part = predictions_table[predictions_table["model"] == name]
        np.testing.assert_array_equal(
            np.sort(part["target_index"].to_numpy()),
            target_index,
        )
        np.testing.assert_allclose(
            part["y_true"].to_numpy(),
            series[part["target_index"].to_numpy()],
        )

    output_dir = base_dir / "results"
    save_experiment_tables(
        tuning_table,
        metrics_table,
        predictions_table,
        output_dir,
    )

    # 从已保存的 CSV 重新读取预测，再计算指标核对
    saved_predictions = pd.read_csv(
        output_dir / "predictions.csv",
        parse_dates=["target_time"],
    )
    saved_metrics = pd.read_csv(output_dir / "metrics.csv")

    for row in saved_metrics.itertuples(index=False):
        part = saved_predictions[
            (saved_predictions["model"] == row.model)
            & (saved_predictions["split"] == row.split)
        ]
        scores = calculate_metrics(part["y_true"], part["y_pred"])
        assert len(part) == row.n_samples
        np.testing.assert_allclose(
            [scores["mae"], scores["rmse"]],
            [row.mae, row.rmse],
            rtol=1e-10,
            atol=1e-10,
        )

    print("\n最终指标：")
    print(metrics_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))
    print("\n预测表行数：", len(saved_predictions))
    print("CSV 保存及指标回算核对通过")
        # 调参图直接读取已保存的 tuning.csv
    saved_tuning = pd.read_csv(output_dir / "tuning.csv")

    plot_experiment(
        df,
        saved_tuning,
        saved_predictions,
        window_size,
        output_dir,
    )
    print("三张图已保存")

    # 打印连续6小时的数据，供 README 的场景分析使用
    local = saved_predictions[
        (saved_predictions["split"] == "val")
        & (saved_predictions["target_time"] >= "2017-06-26 00:00")
        & (saved_predictions["target_time"] < "2017-06-26 06:00")
    ]

    local_table = local.pivot(
        index="target_time", columns="model", values="y_pred"
    )
    local_truth = local[local["model"] == "last_value"].set_index(
        "target_time"
    )["y_true"]
    local_table.insert(0, "真实油温", local_truth)
    local_table["HUFL"] = df.set_index("date").loc[
        local_table.index, "HUFL"
    ]

    print("\n场景分析：验证集开始的连续6小时")
    print(local_table.to_string(float_format=lambda x: f"{x:.6f}"))

if __name__ == "__main__":
    main()
        # 调参图直接读取已保存的 tuning.csv
    