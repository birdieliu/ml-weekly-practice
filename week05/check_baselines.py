from data import prepare_tensor_data
from training import fit_baselines
from evaluation import (
    evaluate_mean_baseline,
    evaluate_linear_baseline,
)

data = prepare_tensor_data()

mean_value, linear_model = fit_baselines(data)

print("均值基线的固定预测：", mean_value)

for split in ("train", "val"):
    X = data[f"X_{split}"]
    y_raw = data[f"y_{split}_raw"]

    mean_metrics = evaluate_mean_baseline(
        mean_value,
        X,
        y_raw,
    )

    linear_metrics = evaluate_linear_baseline(
        linear_model,
        X,
        y_raw,
    )

    print(f"\n{split} 的原始单位指标：")

    print(
        "均值基线：",
        "MAE =", mean_metrics["mae"],
        "RMSE =", mean_metrics["rmse"],
    )

    print(
        "线性回归：",
        "MAE =", linear_metrics["mae"],
        "RMSE =", linear_metrics["rmse"],
    )
    