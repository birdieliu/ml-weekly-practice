import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
        )

        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 不传入初始状态，每次调用使用默认零状态
        output, (h_n, c_n) = self.lstm(x)

        # 取每条样本最后一个时间步的隐藏表示
        last_hidden = output[:, -1, :]

        # 线性层将隐藏表示转换为一个预测值
        prediction = self.head(last_hidden)

        return prediction