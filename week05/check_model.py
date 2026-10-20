import torch

from data import prepare_tensor_data, create_train_loader
from models import create_mlp

data = prepare_tensor_data()
loader = create_train_loader(data)

xb, yb = next(iter(loader))

torch.manual_seed(42)
model = create_mlp(hidden_dim=16)

print("模型结构：")
print(model)

print("\n这一批输入的形状：", xb.shape)
print("这一批答案的形状：", yb.shape)

with torch.no_grad():
    output = xb

    for layer in model:
        output = layer(output)
        print(
            "经过",
            layer.__class__.__name__,
            "后的形状：",
            output.shape,
        )

assert output.shape == yb.shape
print("\n检查通过：预测与答案形状一致")