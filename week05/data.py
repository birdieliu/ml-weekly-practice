import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
def prepare_step_data():
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float32)

    return x, y
def load_regression_data():
    dataset = load_diabetes(as_frame=True, scaled=False)

    X = dataset.data
    y = dataset.target

    return X, y
def split_regression_data():
    X, y = load_regression_data()

    X_train, X_other, y_train, y_other = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_other,
        y_other,
        test_size=0.5,
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
def prepare_scaled_data():
    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_regression_data()
    )

    # 输入：只用训练集计算每一列的均值和标准差
    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    # 保留原始答案，后面计算原始单位的误差时要用
    y_train_raw = y_train.to_numpy()
    y_val_raw = y_val.to_numpy()
    y_test_raw = y_test.to_numpy()

    # 答案：只用训练集计算均值和标准差
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std(ddof=0)

    assert y_std > 0

    y_train_scaled = (y_train_raw - y_mean) / y_std
    y_val_scaled = (y_val_raw - y_mean) / y_std
    y_test_scaled = (y_test_raw - y_mean) / y_std

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "y_test": y_test_scaled,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_scaler": x_scaler,
    }

def prepare_tensor_data():
    data = prepare_scaled_data()

    for split in ("train", "val", "test"):
        data[f"X_{split}_tensor"] = torch.tensor(
            data[f"X_{split}"],
            dtype=torch.float32,
        )

        data[f"y_{split}_tensor"] = torch.tensor(
            data[f"y_{split}"],
            dtype=torch.float32,
        ).reshape(-1, 1)

    return data


def create_train_loader(data, batch_size=32):
    dataset = TensorDataset(
        data["X_train_tensor"],
        data["y_train_tensor"],
    )

    generator = torch.Generator().manual_seed(42)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )

    return loader