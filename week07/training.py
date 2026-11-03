from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .models import LSTMForecaster
else:
    from models import LSTMForecaster


def make_loader(part, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(part["X"]),
        torch.from_numpy(part["y"]),
    )

    # 每次创建独立生成器，两种配置从相同种子开始
    generator = torch.Generator().manual_seed(42)

    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def evaluate_mse(model, loader):
    """按全部目标的平方误差总和计算标准化尺度 MSE。"""
    model.eval()
    squared_error_sum = 0.0
    target_count = 0

    with torch.no_grad():
        for x, y in loader:
            prediction = model(x)
            squared_error_sum += (
                (prediction - y).square().sum().item()
            )
            target_count += y.numel()

    return squared_error_sum / target_count


def fit_one_model(splits, hidden_size):
    # 每组独立初始化模型、优化器和数据加载器
    torch.manual_seed(42)
    model = LSTMForecaster(hidden_size)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0,
    )
    loss_function = nn.MSELoss()

    train_loader = make_loader(splits["train"], shuffle=True)
    train_eval_loader = make_loader(
        splits["train"], shuffle=False
    )
    val_loader = make_loader(splits["val"], shuffle=False)

    history = []
    best_val_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, 41):
        model.train()

        for x, y in train_loader:
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()

        # 使用本轮结束后的模型，重新评价完整训练与验证集
        train_mse = evaluate_mse(model, train_eval_loader)
        val_mse = evaluate_mse(model, val_loader)

        history.append({
            "model": f"lstm_h{hidden_size}",
            "hidden_size": hidden_size,
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })

        # 严格小于才替换，完全相同时保留更早轮次
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        print(
            f"h={hidden_size}, epoch={epoch:02d}/40, "
            f"train_mse={train_mse:.6f}, "
            f"val_mse={val_mse:.6f}",
            flush=True,
        )

    # 训练满40轮后，恢复验证表现最好的参数
    model.load_state_dict(best_state)
    model.eval()

    return model, history, best_epoch, best_val_mse