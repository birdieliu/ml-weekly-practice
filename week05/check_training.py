import torch

from data import prepare_tensor_data, create_train_loader
from models import create_mlp
from optimizers import create_optimizer
from training import train_mlp
from evaluation import evaluate_scaled_mse, evaluate_mlp_raw

data = prepare_tensor_data()
loader = create_train_loader(data, batch_size=32)

torch.manual_seed(42)
model = create_mlp(hidden_dim=16)

optimizer = create_optimizer(
    model,
    name="SGD",
    lr=0.01,
)

history, best_epoch, best_val_mse = train_mlp(
    model,
    optimizer,
    loader,
    data,
    epochs=200,
)

restored_val_mse = evaluate_scaled_mse(
    model,
    data["X_val_tensor"],
    data["y_val_tensor"],
)

print("\n记录的轮数：", len(history))
print("最佳轮次：", best_epoch)
print("最佳验证 MSE：", best_val_mse)
print("恢复参数后的验证 MSE：", restored_val_mse)

assert len(history) == 200
assert abs(restored_val_mse - best_val_mse) < 1e-6

print("检查通过：已训练 200 轮，并恢复最佳参数")

train_metrics = evaluate_mlp_raw(
    model,
    data["X_train_tensor"],
    data["y_train_raw"],
    data["y_mean"],
    data["y_std"],
)

val_metrics = evaluate_mlp_raw(
    model,
    data["X_val_tensor"],
    data["y_val_raw"],
    data["y_mean"],
    data["y_std"],
)

print("\n最佳模型的原始单位指标：")
print("训练 MAE：", train_metrics["mae"])
print("训练 RMSE：", train_metrics["rmse"])
print("验证 MAE：", val_metrics["mae"])
print("验证 RMSE：", val_metrics["rmse"])

expected_val_rmse = (
    best_val_mse ** 0.5
) * data["y_std"]

assert abs(
    val_metrics["rmse"] - expected_val_rmse
) < 1e-4

print("检查通过：原始单位 RMSE 与标准化 MSE 对应")