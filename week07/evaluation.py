import numpy as np
import pandas as pd
import torch


def predict_original(model, part, mu, sigma):
    """分批预测，并还原到原始油温尺度。"""
    model.eval()
    batches = []

    with torch.no_grad():
        for start in range(0, len(part["X"]), 128):
            x = torch.from_numpy(part["X"][start:start + 128])
            prediction = model(x)
            batches.append(prediction.numpy().reshape(-1))

    standardized = np.concatenate(batches)
    return standardized * sigma + mu


def make_prediction_table(model, part, mu, sigma, model_name, split):
    y_pred = predict_original(model, part, mu, sigma)

    return pd.DataFrame({
        "model": model_name,
        "split": split,
        "target_index": part["target_index"],
        "target_time": part["target_time"],
        "y_true": part["y_raw"],
        "y_pred": y_pred,
    })