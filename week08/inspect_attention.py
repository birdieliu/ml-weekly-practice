from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from week07.sequence_data import prepare_data
from week08.models import SingleHeadForecaster


def main():
    output_dir = Path(__file__).resolve().parent / "results"

    checkpoint = torch.load(
        output_dir / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = SingleHeadForecaster(
        d_model=checkpoint["d_model"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits, mu, sigma = prepare_data()

    # 确保观察时与训练时的标准化一致
    np.testing.assert_allclose(
        [mu, sigma],
        [checkpoint["mu"], checkpoint["sigma"]],
    )

    # 固定使用第一条验证样本，保留批次维度
    x = torch.from_numpy(splits["val"]["X"][0:1])
    target_time = str(splits["val"]["target_time"][0])
    target_raw = float(splits["val"]["y_raw"].reshape(-1)[0])

    with torch.no_grad():
        prediction, details = model(x, return_details=True)

        Q = details["Q"]
        K = details["K"]
        V = details["V"]
        scores = details["scores"]
        A = details["A"]
        Z = details["Z"]

        # 核对1：第24小时查询与第3小时键的匹配分数
        score_manual = (
            Q[0, 23] * K[0, 2]
        ).sum() / (K.shape[-1] ** 0.5)

        torch.testing.assert_close(
            score_manual, scores[0, 23, 2]
        )

        # 核对2：沿第24行做softmax，还原该行权重
        weights_manual = torch.softmax(
            scores[0, 23, :], dim=-1
        )
        torch.testing.assert_close(
            weights_manual, A[0, 23, :]
        )

        # 核对3：用末行权重和全部V还原末位向量
        z_manual = (
            A[0, 23, :, None] * V[0]
        ).sum(dim=0)

        torch.testing.assert_close(
            z_manual, Z[0, 23, :],
            rtol=1e-5, atol=1e-6,
        )

        # 核对4：末位向量经过预测层，还原模型预测
        prediction_manual = model.head(z_manual)
        torch.testing.assert_close(
            prediction_manual, prediction[0],
            rtol=1e-5, atol=1e-6,
        )

        # 相同分数经过softmax，应得到均匀权重
        equal_weights = torch.softmax(
            torch.zeros(24), dim=-1
        )
        torch.testing.assert_close(
            equal_weights, torch.full((24,), 1 / 24)
        )

    prediction_raw = prediction.item() * sigma + mu
    score_value = scores[0, 23, 2].item()
    weight_value = A[0, 23, 2].item()
    row_sum = A[0, 23, :].sum().item()
    restore_error = (
        z_manual - Z[0, 23, :]
    ).abs().max().item()

    report = (
        f"目标时间：{target_time}\n"
        f"真实油温：{target_raw:.6f}\n"
        f"预测油温：{prediction_raw:.6f}\n"
        f"scores[0, 23, 2]：{score_value:.8f}\n"
        f"A[0, 23, 2]：{weight_value:.8f}\n"
        f"末行权重之和：{row_sum:.8f}\n"
        f"末位向量还原最大误差：{restore_error:.10f}\n"
        f"相同分数对应的权重：{equal_weights[0].item():.8f}\n"
        "全部数值核对通过\n"
    )

    print(report)
    (output_dir / "window_checks.txt").write_text(
        report, encoding="utf-8"
    )

    # 保存本次前向结果，方便后续复查
    np.savez(
        output_dir / "window_details.npz",
        **{
            name: value.numpy()
            for name, value in details.items()
        },
        prediction=prediction.numpy(),
        target_time=target_time,
    )

    # 两幅图使用相同坐标，各自使用颜色刻度
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), constrained_layout=True
    )

    matrices = [
        scores[0].numpy(),
        A[0].numpy(),
    ]
    titles = [
        "Scaled scores S",
        "Attention weights A",
    ]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, origin="upper", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(title)
        ax.set_xlabel("Key hour (being queried)")
        ax.set_ylabel("Query hour")

        ticks = np.arange(24)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks + 1, fontsize=7)
        ax.set_yticklabels(ticks + 1, fontsize=7)

        # 图上第24行、第3列，对应数组下标23、2
        ax.scatter(
            [2], [23],
            s=140, facecolors="none",
            edgecolors="red", linewidths=1.5,
        )

    fig.suptitle(
        f"Target: {target_time}\n"
        f"S[24,3]={score_value:.6f}, "
        f"A[24,3]={weight_value:.6f}, "
        f"last-row sum={row_sum:.6f}"
    )

    fig.savefig(
        output_dir / "scores_attention.png", dpi=180
    )
    plt.close(fig)


if __name__ == "__main__":
    main()