from data import prepare_step_data
from models import create_step_model
from optimizers import create_optimizer
from training import calculate_step_loss, update_step

x, y = prepare_step_data()
model = create_step_model()
optimizer = create_optimizer(model, name="SGD", lr=0.1)

print("更新前的权重：", model.weight.item())
print("更新前的偏置：", model.bias.item())

old_pred, old_loss = update_step(model, x, y, optimizer)

print("更新后的权重：", model.weight.item())
print("更新后的偏置：", model.bias.item())

print("原来保存的预测：")
print(old_pred)
print("原来保存的损失：", old_loss.item())

new_pred, new_loss = calculate_step_loss(model, x, y)

print("重新计算的预测：")
print(new_pred)
print("重新计算的损失：", new_loss.item())