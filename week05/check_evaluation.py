import torch

from data import prepare_tensor_data
from models import create_mlp
from optimizers import create_optimizer
from evaluation import evaluate_scaled_mse

data = prepare_tensor_data()

torch.manual_seed(42)
model = create_mlp(hidden_dim=16)

sgd = create_optimizer(model, name="SGD", lr=0.01)
adam = create_optimizer(model, name="Adam", lr=0.001)

print("SGD 学习率：", sgd.param_groups[0]["lr"])
print("Adam 学习率：", adam.param_groups[0]["lr"])

parameters_before = [
    parameter.detach().clone()
    for parameter in model.parameters()
]

train_mse = evaluate_scaled_mse(
    model,
    data["X_train_tensor"],
    data["y_train_tensor"],
)

val_mse = evaluate_scaled_mse(
    model,
    data["X_val_tensor"],
    data["y_val_tensor"],
)

print("未训练模型的训练 MSE：", train_mse)
print("未训练模型的验证 MSE：", val_mse)

for before, after in zip(parameters_before, model.parameters()):
    assert torch.equal(before, after.detach())

print("检查通过：评价前后的模型参数完全相同")