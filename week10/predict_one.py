from pathlib import Path

import torch

from week07.sequence_data import prepare_data
from week10.encoder import EncoderForecaster


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = (
        Path(__file__).resolve().parent / "results" / "best_encoder.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = EncoderForecaster(
        d_model=checkpoint["d_model"],
        d_head=checkpoint["d_head"],
        d_ff=checkpoint["d_ff"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits, mu, sigma = prepare_data()
    x = torch.from_numpy(splits["val"]["X"][0:1])
    target_raw = float(splits["val"]["y_raw"][0].reshape(-1)[0])
    target_time = str(splits["val"]["target_time"][0])

    with torch.no_grad():
        prediction, details = model(x, return_details=True)

    prediction_raw = prediction.item() * sigma + mu

    print(f"载入模型：{checkpoint_path}")
    print(f"最佳轮次：{checkpoint['best_epoch']}")
    print(f"目标时间：{target_time}")
    print(f"编码器输出 E 形状：{tuple(details['E'].shape)}")
    print(f"送入预测层的末行形状：{tuple(details['last'].shape)}")
    print(f"预测形状：{tuple(prediction.shape)}")
    print(f"预测油温：{prediction_raw:.6f}")
    print(f"真实油温：{target_raw:.6f}")


if __name__ == "__main__":
    main()