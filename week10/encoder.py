import torch
from torch import nn

from week09.position import add_position
from week10.attention import MultiHeadAttention


class EncoderForecaster(nn.Module):
    def __init__(self, d_model=16, d_head=8, d_ff=32):
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)
        self.attention = MultiHeadAttention(d_model, d_head)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, x, return_details=False):
        # x: (B, 24, 1)，先映射到16维，再加位置向量
        embedded = self.input_proj(x)
        X = add_position(embedded)  # (B, 24, 16)

        # 油温预测时，24个输入小时都已观测到，因此关闭因果遮罩
        M, attention_weights = self.attention(
            X, use_causal_mask=False
        )  # M: (B, 24, 16)

        H = self.norm1(X + M)  # 第一次残差相加与归一化
        F = self.ff2(self.activation(self.ff1(H)))  # (B, 24, 16)
        E = self.norm2(H + F)  # 第二次残差相加与归一化

        # 编码器处理完全部24行后，才取第24行预测第25小时
        last = E[:, 23, :]  # (B, 16)
        prediction = self.prediction_head(last)  # (B, 1)

        if return_details:
            return prediction, {
                "X": X,
                "M": M,
                "H": H,
                "F": F,
                "E": E,
                "last": last,
                "attention_weights": attention_weights,
            }

        return prediction