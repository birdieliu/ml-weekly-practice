
import copy

from evaluation import evaluate_scaled_mse

from torch import nn

from sklearn.linear_model import LinearRegression

def calculate_step_loss(model, x, y):
    loss_fn = nn.MSELoss()

    pred = model(x)
    assert pred.shape == y.shape

    loss = loss_fn(pred, y)

    return pred, loss


def record_state(model, stage, loss=None):
    weight_grad = model.weight.grad
    bias_grad = model.bias.grad

    return {
        "stage": stage,
        "weight": model.weight.item(),
        "bias": model.bias.item(),
        "weight_grad": (
            None if weight_grad is None else weight_grad.item()
        ),
        "bias_grad": (
            None if bias_grad is None else bias_grad.item()
        ),
        "loss": loss,
    }


def update_step(model, x, y, optimizer):
    optimizer.zero_grad(set_to_none=True)

    pred, loss = calculate_step_loss(model, x, y)

    records = [
        record_state(model, "after_zero_grad", loss.item())
    ]

    loss.backward()

    records.append(
        record_state(model, "after_backward")
    )

    optimizer.step()

    _, new_loss = calculate_step_loss(model, x, y)

    records.append(
        record_state(model, "after_step", new_loss.item())
    )

    return pred, loss, records

def train_mlp(model, optimizer, loader, data, epochs=200):
    loss_fn = nn.MSELoss()

    history = []

    best_val_mse = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            assert pred.shape == yb.shape

            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

        train_mse = evaluate_scaled_mse(
            model,
            data["X_train_tensor"],
            data["y_train_tensor"],
        )

        val_mse = evaluate_scaled_mse(
            model,
            data["X_val_tensor"],
            data["y_val_tensor"],
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"第 {epoch} 轮："
                f"训练 MSE={train_mse:.4f}，"
                f"验证 MSE={val_mse:.4f}"
            )

    model.load_state_dict(best_state)

    return history, best_epoch, best_val_mse

def fit_baselines(data):
    mean_value = float(data["y_train_raw"].mean())

    linear_model = LinearRegression()
    linear_model.fit(
        data["X_train"],
        data["y_train_raw"],
    )

    return mean_value, linear_model