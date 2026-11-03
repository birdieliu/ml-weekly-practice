import torch

from sequence_data import prepare_data
from models import LSTMForecaster


def main():
    splits, mu, sigma = prepare_data()

    # 两种配置使用同一批实际样本
    x = torch.from_numpy(splits["train"]["X"][:4])

    for hidden_size in (16, 32):
        torch.manual_seed(42)
        model = LSTMForecaster(hidden_size)
        model.eval()

        with torch.no_grad():
            output, (h_n, c_n) = model.lstm(x)
            last_hidden = output[:, -1, :]
            prediction = model(x)

            # 单层、单向时，这两个表示应相同
            torch.testing.assert_close(last_hidden, h_n[0])

            # 核对模型确实使用末步表示生成预测
            torch.testing.assert_close(
                prediction,
                model.head(last_hidden),
            )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"\nhidden_size={hidden_size}")
        print("输入 x：", tuple(x.shape))
        print("每步隐藏输出 output：", tuple(output.shape))
        print("最终隐藏状态 h_n：", tuple(h_n.shape))
        print("最终细胞状态 c_n：", tuple(c_n.shape))
        print("末步表示 last_hidden：", tuple(last_hidden.shape))
        print("预测 prediction：", tuple(prediction.shape))
        print("模型参数总数：", parameter_count)
        print("末步表示与 h_n[0] 核对通过")


if __name__ == "__main__":
    main()