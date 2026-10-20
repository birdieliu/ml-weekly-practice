import torch
from torch import nn


def create_step_model():
    model = nn.Linear(1, 1)

    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    return model

#实践二
def create_mlp(hidden_dim=16):
    model = nn.Sequential(
        nn.Linear(10, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

    return model