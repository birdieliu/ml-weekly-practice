import numpy as np
import torch
from torch import nn


def evaluate_scaled_mse(model, X, y):
    model.eval()

    with torch.no_grad():
        pred = model(X)
        assert pred.shape == y.shape

        loss_fn = nn.MSELoss()
        mse = loss_fn(pred, y).item()

    return mse

def calculate_raw_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    assert y_true.shape == y_pred.shape

    errors = y_pred - y_true

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def evaluate_mlp_raw(model, X, y_raw, y_mean, y_std):
    model.eval()

    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy().reshape(-1)

    pred_raw = pred_scaled * y_std + y_mean

    return calculate_raw_metrics(y_raw, pred_raw)

def evaluate_mean_baseline(mean_value, X, y_raw):
    pred_raw = np.full(
        shape=len(X),
        fill_value=mean_value,
        dtype=float,
    )

    return calculate_raw_metrics(y_raw, pred_raw)


def evaluate_linear_baseline(linear_model, X, y_raw):
    pred_raw = linear_model.predict(X)

    return calculate_raw_metrics(y_raw, pred_raw)