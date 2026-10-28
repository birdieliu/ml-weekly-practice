import numpy as np


def calculate_metrics(y_true, y_pred):
    """计算原始油温尺度上的 MAE 和 RMSE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("真实值和预测值必须是一维数组")

    if y_true.shape != y_pred.shape:
        raise ValueError("真实值与预测值的形状不一致")

    if y_true.size == 0:
        raise ValueError("不能评价空数组")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("真实值或预测值中存在无效数值")

    errors = y_pred - y_true

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }