import math

from week09.position import add_position

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week07.sequence_data import prepare_data
from week08.models import SingleHeadForecaster


def forward_with_position(model, x):
    embedded = model.input_proj(x)

    # 先加入位置，再生成Q、K、V
    embedded = add_position(embedded)

    Q = model.q_proj(embedded)
    K = model.k_proj(embedded)
    V = model.v_proj(embedded)

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(K.shape[-1])

    A = torch.softmax(scores, dim=-1)
    Z = A @ V

    prediction = model.head(Z[:, -1, :])

    return prediction, {
        "Q": Q,
        "K": K,
        "V": V,
        "A": A,
        "Z": Z,
    }

def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "week09" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        root / "week08" / "results" / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = SingleHeadForecaster(checkpoint["d_model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits, mu, sigma = prepare_data()
    np.testing.assert_allclose(
        [mu, sigma],
        [checkpoint["mu"], checkpoint["sigma"]],
    )

    # 固定使用第一条验证窗口
    x = torch.from_numpy(splits["val"]["X"][0:1])

    # 从前23小时中选最低、最高油温，确保数值不同
    i = int(x[0, :23, 0].argmin().item())
    j = int(x[0, :23, 0].argmax().item())
    assert x[0, i, 0].item() != x[0, j, 0].item()

    swapped = x.clone()
    swapped[:, i, :] = x[:, j, :]
    swapped[:, j, :] = x[:, i, :]

    target_time = pd.Timestamp(
        splits["val"]["target_time"][0]
    )

    for index in (i, j):
        time = target_time - pd.Timedelta(hours=24 - index)
        temperature = x[0, index, 0].item() * sigma + mu
        print(
            f"交换位置：第{index + 1}小时，"
            f"时间：{time}，油温：{temperature:.6f}"
        )

    with torch.no_grad():
        p_before, before = model(x, return_details=True)
        p_after, after = model(swapped, return_details=True)

        # 核对1：末位查询没有变化
        torch.testing.assert_close(
            before["Q"][:, -1, :],
            after["Q"][:, -1, :],
        )

        # 核对2：末行权重随交换位置重新排列
        expected_weights = before["A"][:, -1, :].clone()
        expected_weights[:, i] = before["A"][:, -1, j]
        expected_weights[:, j] = before["A"][:, -1, i]

        torch.testing.assert_close(
            after["A"][:, -1, :],
            expected_weights,
            rtol=1e-5, atol=1e-6,
        )

        # 核对3：末位汇总向量相同
        torch.testing.assert_close(
            before["Z"][:, -1, :],
            after["Z"][:, -1, :],
            rtol=1e-5, atol=1e-6,
        )

        # 核对4：最终预测相同
        torch.testing.assert_close(
            p_before, p_after,
            rtol=1e-5, atol=1e-6,
        )

    before_raw = p_before.item() * sigma + mu
    after_raw = p_after.item() * sigma + mu

    print(f"\n交换前预测：{before_raw:.8f}")
    print(f"交换后预测：{after_raw:.8f}")
    print(f"预测绝对差：{abs(after_raw - before_raw):.10f}")
    print(
        "末位Z最大分量差："
        f"{(before['Z'][:, -1, :] - after['Z'][:, -1, :]).abs().max().item():.10f}"
    )
    print("未加位置：全部核对通过")

    pd.DataFrame([{
        "position_encoding": False,
        "swapped_hour_i": i + 1,
        "swapped_hour_j": j + 1,
        "prediction_before": before_raw,
        "prediction_after": after_raw,
        "prediction_difference": abs(after_raw - before_raw),
    }]).to_csv(
        output_dir / "swap_without_position.csv",
        index=False,
    )

        # 对同样的原窗口和交换窗口，加入相同的位置编号
    with torch.no_grad():
        pos_p_before, pos_before = forward_with_position(model, x)
        pos_p_after, pos_after = forward_with_position(model, swapped)

        # 最后一个小时的内容和位置都没变，查询仍应相同
        torch.testing.assert_close(
            pos_before["Q"][:, -1, :],
            pos_after["Q"][:, -1, :],
        )

    pos_before_raw = pos_p_before.item() * sigma + mu
    pos_after_raw = pos_p_after.item() * sigma + mu

    print("\n加入位置后：")
    print(f"交换前预测：{pos_before_raw:.8f}")
    print(f"交换后预测：{pos_after_raw:.8f}")
    print(
        f"预测绝对差："
        f"{abs(pos_after_raw - pos_before_raw):.10f}"
    )
    print(
        "末位Z最大分量差："
        f"{(pos_before['Z'][:, -1, :] - pos_after['Z'][:, -1, :]).abs().max().item():.10f}"
    )

    table = pd.DataFrame([
        {
            "position_encoding": False,
            "swapped_hour_i": i + 1,
            "swapped_hour_j": j + 1,
            "prediction_before": before_raw,
            "prediction_after": after_raw,
            "prediction_difference": abs(after_raw - before_raw),
        },
        {
            "position_encoding": True,
            "swapped_hour_i": i + 1,
            "swapped_hour_j": j + 1,
            "prediction_before": pos_before_raw,
            "prediction_after": pos_after_raw,
            "prediction_difference": abs(
                pos_after_raw - pos_before_raw
            ),
        },
    ])

    table.to_csv(
        output_dir / "swap_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()