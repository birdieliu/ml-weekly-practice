from pathlib import Path

import numpy as np
import torch

from week07.sequence_data import prepare_data
from week08.models import SingleHeadForecaster
from week10.attention import MultiHeadAttention


def max_difference(a, b):
    return (a - b).abs().max().item()


def inspect_case(model, x_original, x_changed, use_causal_mask):
    # 同一模型参数，分别计算原窗口和改动后的窗口
    with torch.no_grad():
        output_original, (a1_original, _) = model(
            x_original, use_causal_mask=use_causal_mask
        )
        output_changed, _ = model(
            x_changed, use_causal_mask=use_causal_mask
        )

    # 第 6 小时是 Python 下标 5；第 18 小时是下标 17
    row_original = output_original[0, 5, :]
    row_changed = output_changed[0, 5, :]
    row_diff = max_difference(row_original, row_changed)

    first_head_row = a1_original[0, 5, :]
    future_weights = first_head_row[6:]  # 第 7 到第 24 小时

    return {
        "row_diff": row_diff,
        "weight_18": first_head_row[17].item(),
        "future_max": future_weights.max().item(),
        "weight_sum": first_head_row.sum().item(),
    }


def main():
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]

    # 第八周的最佳模型输入映射和同一条验证窗口
    checkpoint = torch.load(
        root / "week08" / "results" / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )
    input_model = SingleHeadForecaster(d_model=checkpoint["d_model"])
    input_model.load_state_dict(checkpoint["model_state_dict"])
    input_model.eval()

    splits, _, _ = prepare_data()
    x_raw = torch.from_numpy(splits["val"]["X"][0:1])
    target_time = str(splits["val"]["target_time"][0])

    # 只改副本的第 18 小时：加 1 个训练集标准差
    x_changed_raw = x_raw.clone()
    x_changed_raw[0, 17, 0] += 1.0

    with torch.no_grad():
        x_original = input_model.input_proj(x_raw)
        x_changed = input_model.input_proj(x_changed_raw)

    assert x_original.shape == (1, 24, 16)
    assert x_changed.shape == (1, 24, 16)

    # 两种遮罩设置复用同一个模型、同一组头参数
    model = MultiHeadAttention(d_model=16, d_head=8)
    model.eval()

    unmasked = inspect_case(
        model, x_original, x_changed, use_causal_mask=False
    )
    masked = inspect_case(
        model, x_original, x_changed, use_causal_mask=True
    )

    report = (
        f"目标时间：{target_time}\n"
        f"改动：第 18 小时标准化油温加 1\n"
        f"未遮罩：第 6 行输出最大差值={unmasked['row_diff']:.10f}, "
        f"A[0, 5, 17]={unmasked['weight_18']:.10f}\n"
        f"未遮罩：第一头未来列最大权重={unmasked['future_max']:.10f}, "
        f"第 6 行权重和={unmasked['weight_sum']:.10f}\n"
        f"已遮罩：第 6 行输出最大差值={masked['row_diff']:.10f}, "
        f"A[0, 5, 17]={masked['weight_18']:.10f}\n"
        f"已遮罩：第一头未来列最大权重={masked['future_max']:.10f}, "
        f"第 6 行权重和={masked['weight_sum']:.10f}\n"
    )

    print(report)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "practice5_checks.txt").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()