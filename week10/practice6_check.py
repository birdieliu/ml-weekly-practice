import torch

from week07.sequence_data import prepare_data
from week10.encoder import EncoderForecaster


def main():
    splits, _, _ = prepare_data()

    x = torch.from_numpy(splits["val"]["X"][0:1])
    y = torch.from_numpy(splits["val"]["y"][0:1])

    model = EncoderForecaster()
    model.train()

    prediction, details = model(x, return_details=True)
    loss = torch.nn.functional.mse_loss(prediction, y)

    print(f"输入 x：{tuple(x.shape)}")
    print(f"注意力输出 M：{tuple(details['M'].shape)}")
    print(f"第一次残差输入 H：{tuple(details['H'].shape)}")
    print(f"前馈输出 F：{tuple(details['F'].shape)}")
    print(f"编码器输出 E：{tuple(details['E'].shape)}")
    print(f"编码器末行：{tuple(details['last'].shape)}")
    print(f"预测值：{tuple(prediction.shape)}")
    print(f"目标值：{tuple(y.shape)}")

    loss.backward()

    q_grad = model.attention.head1.q_proj.weight.grad
    prediction_grad = model.prediction_head.weight.grad

    print(f"第一头 Q 投影有梯度：{q_grad is not None}")
    print(f"第一头 Q 投影梯度形状：{tuple(q_grad.shape)}")
    print(f"预测层有梯度：{prediction_grad is not None}")
    print(f"预测层梯度形状：{tuple(prediction_grad.shape)}")


if __name__ == "__main__":
    main()