from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week07.sequence_data import prepare_data
from week07.training import make_loader
from week08.models import SingleHeadForecaster


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

    part = splits["val"]
    loader = make_loader(part, shuffle=False)
    predictions = {
        "original": [],
        "mean": [],
        "last": [],
    }

    with torch.no_grad():
        for x, _ in loader:
            prediction, details = model(
                x, return_details=True
            )

            V = details["V"]             # (B, 24, 16)
            weights = details["A"][:, -1, :]  # (B, 24)

            # 对每个窗口，分别汇总它自己的V
            z_original = (
                weights.unsqueeze(-1) * V
            ).sum(dim=1)

            z_mean = V.mean(dim=1)
            z_last = V[:, -1, :]

            p_original = model.head(z_original)

            # 确认自己的汇总代码还原了原模型预测
            torch.testing.assert_close(
                p_original, prediction,
                rtol=1e-5, atol=1e-6,
            )

            predictions["original"].append(p_original)
            predictions["mean"].append(model.head(z_mean))
            predictions["last"].append(model.head(z_last))

    targets = part["y_raw"].reshape(-1)
    target_times = part["target_time"]
    rows = []
    prediction_table = pd.DataFrame({
        "target_time": target_times,
        "actual": targets,
    })

    for mode, batches in predictions.items():
        predicted = (
            torch.cat(batches).numpy().reshape(-1)
            * sigma + mu
        )

        assert len(predicted) == len(targets)
        prediction_table[mode] = predicted

        error = predicted - targets
        rows.append({
            "mode": mode,
            "val_samples": len(targets),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        })

    table = pd.DataFrame(rows)

    table.to_csv(
        output_dir / "reading_metrics.csv",
        index=False,
    )
    prediction_table.to_csv(
        output_dir / "reading_predictions.csv",
        index=False,
    )

    print(table.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()