import math

import torch
from torch import nn


class SingleHeadForecaster(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()

        # 每小时的一个油温 → 16 维向量
        self.input_proj = nn.Linear(1, d_model)

        # 分别生成查询、键和值
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 读取历史后得到的向量 → 一个预测油温
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_details=False):
        # x: (B, 24, 1)
        embedded = self.input_proj(x)

        Q = self.q_proj(embedded)
        K = self.k_proj(embedded)
        V = self.v_proj(embedded)
        # Q、K、V: (B, 24, 16)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(K.shape[-1])
        # scores: (B, 24, 24)

        A = torch.softmax(scores, dim=-1)
        Z = A @ V
        # A: (B, 24, 24)
        # Z: (B, 24, 16)

        prediction = self.head(Z[:, -1, :])
        # prediction: (B, 1)

        if return_details:
            return prediction, {
                "Q": Q,
                "K": K,
                "V": V,
                "scores": scores,
                "A": A,
                "Z": Z,
            }

        return prediction