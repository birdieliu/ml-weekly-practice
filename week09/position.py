import torch


def add_position(embedded):
    # embedded形状：(B, T, D)
    length = embedded.shape[1]

    if length < 2:
        raise ValueError("窗口长度至少为2")

    position = torch.zeros_like(embedded)

    # T个位置从0到1，只放入第一个特征分量
    position[:, :, 0] = torch.linspace(
        0.0,
        1.0,
        steps=length,
        device=embedded.device,
        dtype=embedded.dtype,
    )

    return embedded + position