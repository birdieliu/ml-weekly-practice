from pathlib import Path

import pandas as pd
import torch

from week07.sequence_data import prepare_data
from week08.models import SingleHeadForecaster


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "week09" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载实践一的最佳模型
    checkpoint = torch.load(
        root / "week08" / "results" / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = SingleHeadForecaster(
        d_model=checkpoint["d_model"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits, mu, sigma = prepare_data()

    torch.testing.assert_close(
        torch.tensor([mu, sigma]),
        torch.tensor([checkpoint["mu"], checkpoint["sigma"]]),
    )

    # 仍然使用第一条验证样本，形状为(1, 24, 1)
    x = torch.from_numpy(splits["val"]["X"][0:1])
    target_time = str(splits["val"]["target_time"][0])
    actual = float(splits["val"]["y_raw"].reshape(-1)[0])

    with torch.no_grad():
        prediction, details = model(x, return_details=True)

        # 去掉批次维度，得到24个V和一行24个权重
        V = details["V"][0]                 # (24, 16)
        original_weights = details["A"][0, -1, :].clone()

        # 另建两组权重，保留原模型权重
        mean_weights = torch.full_like(
            original_weights, 1.0 / V.shape[0]
        )

        last_weights = torch.zeros_like(original_weights)
        last_weights[-1] = 1.0

        # 三种权重分别汇总同一组V
        z_original = original_weights @ V
        z_mean = mean_weights @ V
        z_last = last_weights @ V

        # 三种汇总结果经过同一个预测层
        p_original = model.head(z_original)
        p_mean = model.head(z_mean)
        p_last = model.head(z_last)

        # 核对1：原权重能还原模型的汇总向量和预测
        torch.testing.assert_close(
            z_original, details["Z"][0, -1, :],
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            p_original, prediction[0],
            rtol=1e-5, atol=1e-6,
        )

        # 核对2：平均读等于对V沿时间轴求平均
        torch.testing.assert_close(
            z_mean, V.mean(dim=0),
            rtol=1e-5, atol=1e-6,
        )

        # 核对3：只读末小时等于最后一行V
        torch.testing.assert_close(
            z_last, V[-1],
            rtol=1e-5, atol=1e-6,
        )

    rows = []

    for name, z, p in [
        ("original", z_original, p_original),
        ("mean", z_mean, p_mean),
        ("last", z_last, p_last),
    ]:
        # 把标准化预测还原为原始油温
        predicted = p.item() * sigma + mu
        original_raw = p_original.item() * sigma + mu

        rows.append({
            "mode": name,
            "target_time": target_time,
            "actual": actual,
            "prediction": predicted,
            "abs_error": abs(predicted - actual),
            "prediction_change": abs(predicted - original_raw),
            "z_max_change": (z - z_original).abs().max().item(),
        })

    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "replacement_window.csv",
        index=False,
    )

    print("三项程序核对全部通过")
    print(
        f"原模型给末小时的权重："
        f"{original_weights[-1].item():.8f}"
    )
    print(
        f"末小时V与平均V的最大分量差："
        f"{(V[-1] - V.mean(dim=0)).abs().max().item():.8f}"
    )
    print()
    print(table.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()