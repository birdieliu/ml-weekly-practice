from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from week07.sequence_data import prepare_data
from week08.models import SingleHeadForecaster
from week10.attention import AttentionHead


def max_difference(a, b):
    return (a - b).abs().max().item()


def main():
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # 读取第八周模型，复用它的油温输入映射
    checkpoint = torch.load(
        root / "week08" / "results" / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )
    input_model = SingleHeadForecaster(d_model=checkpoint["d_model"])
    input_model.load_state_dict(checkpoint["model_state_dict"])
    input_model.eval()

    # 使用第八周固定的第一条验证窗口，形状 (1, 24, 1)
    splits, mu, sigma = prepare_data()
    x_raw = torch.from_numpy(splits["val"]["X"][0:1])
    target_time = str(splits["val"]["target_time"][0])

    # 第八周的 input_proj 把每小时油温映射到 16 维
    with torch.no_grad():
        x = input_model.input_proj(x_raw)

    assert x.shape == (1, 24, 16)

    # 建立两个头；同一输入 x 分别送入每个头
    head1 = AttentionHead(d_model=16, d_head=8)
    head2 = AttentionHead(d_model=16, d_head=8)

    # 对照一：复制全部参数，验证相同头会产生相同结果
    head2.load_state_dict(head1.state_dict())
    head1.eval()
    head2.eval()

    with torch.no_grad():
        z1_same, a1_same = head1(x)
        z2_same, a2_same = head2(x)

    same_attention_diff = max_difference(
        a1_same[0, 23, :], a2_same[0, 23, :]
    )
    same_output_diff = max_difference(
        z1_same[0, 23, :], z2_same[0, 23, :]
    )

    # 对照二：保留头 1，只重新初始化头 2
    head2 = AttentionHead(d_model=16, d_head=8)
    head2.eval()

    with torch.no_grad():
        z1_diff, a1_diff = head1(x)
        z2_diff, a2_diff = head2(x)

    different_attention_diff = max_difference(
        a1_diff[0, 23, :], a2_diff[0, 23, :]
    )
    different_output_diff = max_difference(
        z1_diff[0, 23, :], z2_diff[0, 23, :]
    )

    # 拼接两个头的末位读出，检查维度如何恢复到 16
    joined = torch.cat(
        [z1_diff[:, 23, :], z2_diff[:, 23, :]], dim=-1
    )
    assert joined.shape == (1, 16)

    output_projection = torch.nn.Linear(16, 16)
    projected = output_projection(joined)
    assert projected.shape == (1, 16)

    report = (
        f"目标时间：{target_time}\n"
        f"输入形状：{tuple(x.shape)}\n"
        f"每头读出形状：{tuple(z1_diff.shape)}\n"
        f"复制参数后，末行权重最大差值：{same_attention_diff:.10f}\n"
        f"复制参数后，末位读出最大差值：{same_output_diff:.10f}\n"
        f"重新初始化第二头后，末行权重最大差值："
        f"{different_attention_diff:.10f}\n"
        f"重新初始化第二头后，末位读出最大差值："
        f"{different_output_diff:.10f}\n"
        f"拼接后的末位读出形状：{tuple(joined.shape)}\n"
        f"输出投影后的末位形状：{tuple(projected.shape)}\n"
    )
    print(report)
    (results_dir / "practice4_checks.txt").write_text(
        report, encoding="utf-8"
    )

    # 并排画出两组对照的末行权重，横轴刻度一致
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    panels = [
        (
            axes[0],
            a1_same[0, 23, :].numpy(),
            a2_same[0, 23, :].numpy(),
            "Copied parameters",
        ),
        (
            axes[1],
            a1_diff[0, 23, :].numpy(),
            a2_diff[0, 23, :].numpy(),
            "Second head reinitialized",
        ),
    ]

    for ax, weights1, weights2, title in panels:
        hours = np.arange(1, 25)
        ax.plot(hours, weights1, marker="o", label="Head 1")
        ax.plot(hours, weights2, marker="x", label="Head 2")
        ax.set_title(title)
        ax.set_xlabel("Hour in input window")
        ax.set_ylabel("Attention weight")
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlim(1, 24)
        ax.legend()

    fig.suptitle(f"Target time: {target_time}")
    fig.savefig(results_dir / "practice4_attention.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()