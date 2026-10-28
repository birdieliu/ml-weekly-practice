import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_last_value(X):
    """用窗口中最近一次观测作为预测。"""
    return np.asarray(X)[:, -1]


def predict_window_mean(X):
    """用每条窗口的平均油温作为预测。"""
    return np.asarray(X).mean(axis=1)


def create_lag_linear():
    """创建标准化与线性回归管道，尚未拟合。"""
    return make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )
def create_ridge(alpha):
    """创建标准化与 Ridge 回归管道。"""
    from sklearn.linear_model import Ridge

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )