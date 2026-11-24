import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, d_model=16, d_head=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head)
        self.k_proj = nn.Linear(d_model, d_head)
        self.v_proj = nn.Linear(d_model, d_head)

    def forward(self, x, use_causal_mask=False):
        # x: (B, T, 16)
        q = self.q_proj(x)  # (B, T, 8)
        k = self.k_proj(x)  # (B, T, 8)
        v = self.v_proj(x)  # (B, T, 8)

        scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        # scores: (B, T, T)，每行对应一个查询位置，每列对应一个被读取位置

        if use_causal_mask:
            T = scores.size(-1)
            future = torch.ones(
                T, T, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(future, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v  # (B, T, 8)
        return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=16, d_head=8):
        super().__init__()
        self.head1 = AttentionHead(d_model, d_head)
        self.head2 = AttentionHead(d_model, d_head)
        self.out_proj = nn.Linear(2 * d_head, d_model)

    def forward(self, x, use_causal_mask=False):
        z1, a1 = self.head1(x, use_causal_mask)
        z2, a2 = self.head2(x, use_causal_mask)

        # 拼接特征维：两个 (B, T, 8) 合成 (B, T, 16)
        joined = torch.cat([z1, z2], dim=-1)
        output = self.out_proj(joined)  # (B, T, 16)
        return output, (a1, a2)