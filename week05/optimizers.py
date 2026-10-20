import torch


def create_optimizer(model, name="SGD", lr=0.1):
    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0,
            weight_decay=0,
        )

    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=0,
        )

    raise ValueError(f"不支持的优化器：{name}")