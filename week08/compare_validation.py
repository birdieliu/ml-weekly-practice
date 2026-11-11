from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week07.models import LSTMForecaster
from week07.sequence_data import prepare_data
from week07.training import make_loader
from week08.models import SingleHeadForecaster


def evaluate(model, loader, targets, mu, sigma):
    model.eval()
    predictions = []

    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(x))

    prediction_raw = (
        torch.cat(predictions).numpy().reshape(-1) * sigma + mu
    )
    error = prediction_raw - targets

    return {
        "val_samples": len(targets),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "week08" / "results"

    splits, mu, sigma = prepare_data()
    loader = make_loader(splits["val"], shuffle=False)
    targets = splits["val"]["y_raw"].reshape(-1)

    lstm_checkpoint = torch.load(
        root / "week07" / "results" / "chosen_lstm.pt",
        map_location="cpu",
        weights_only=True,
    )
    attention_checkpoint = torch.load(
        output_dir / "best_attention.pt",
        map_location="cpu",
        weights_only=True,
    )

    for checkpoint in (lstm_checkpoint, attention_checkpoint):
        np.testing.assert_allclose(
            [mu, sigma],
            [checkpoint["mu"], checkpoint["sigma"]],
        )

    lstm = LSTMForecaster(lstm_checkpoint["hidden_size"])
    lstm.load_state_dict(lstm_checkpoint["model_state_dict"])

    attention = SingleHeadForecaster(
        attention_checkpoint["d_model"]
    )
    attention.load_state_dict(
        attention_checkpoint["model_state_dict"]
    )

    rows = []
    for name, model in [
        ("week07_lstm", lstm),
        ("week08_attention", attention),
    ]:
        rows.append({
            "model": name,
            **evaluate(model, loader, targets, mu, sigma),
        })

    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "validation_comparison.csv", index=False
    )
    print(table.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()