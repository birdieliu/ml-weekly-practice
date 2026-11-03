from pathlib import Path

import pandas as pd
import torch

def forward_sequence(inputs, order, wx, wh, b, source_rows, case):
    h = torch.tensor(0.0)
    records = []

    for step, source_index in enumerate(order, start=1):
        z = inputs[source_index]
        h_prev = h

        input_term = wx * z
        memory_term = wh * h_prev
        h = torch.tanh(input_term + memory_term + b)

        records.append({
            "case": case,
            "step": step,
            "source_index": source_index,
            "source_time": source_rows.iloc[source_index]["date"],
            "z": z.item(),
            "h_prev": h_prev.item(),
            "input_term": input_term.item(),
            "memory_term": memory_term.item(),
            "h": h.item(),
            "input_grad": None,
        })

    prediction = 0.8 * h + 0.1
    return prediction, records

def main():
    # 根据当前文件的位置寻找数据，不依赖终端所在目录
    data_path = (
        Path(__file__).resolve().parents[1]
        / "week06"
        / "data"
        / "ETTh1.csv"
    )

    df = pd.read_csv(data_path)
    first_six = df.loc[:5, ["date", "OT"]].copy()

    # 先用普通数据完成变换
    u = first_six["OT"].to_numpy(dtype="float32")
    z_values = (u - u[0]) / 10.0

    # 前五个值组成一个独立的、可求导的输入张量
    inputs = torch.tensor(
        z_values[:5],
        dtype=torch.float32,
        requires_grad=True,
    )

    # 第六个值作为目标，不需要对目标求梯度
    target = torch.tensor(
        z_values[5],
        dtype=torch.float32,
    )

    print("原始数据：")
    print(first_six.to_string(index=True))

    print("\n变换后的六个值：")
    print(z_values)

    print("\n五步输入：", inputs)
    print("输入形状：", inputs.shape)
    print("输入是否需要梯度：", inputs.requires_grad)
    print("输入是否为叶子张量：", inputs.is_leaf)

    print("\n预测目标：", target.item())
    print("目标形状：", target.shape)
        # 只创建一组参数，五个时间步共同使用
    wx = torch.tensor(0.6, requires_grad=True)
    wh = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

       # Python 下标从0开始
    original_order = [0, 1, 2, 3, 4]
    reordered_order = [0, 3, 2, 1, 4]

    # 原顺序：前向 + 一次反向传播
    prediction, original_records = forward_sequence(
        inputs, original_order, wx, wh, b,
        first_six, case="original",
    )
    loss = (prediction - target) ** 2
    loss.backward()

    for record in original_records:
        index = record["source_index"]
        record["input_grad"] = inputs.grad[index].item()

    print(f"\n原顺序预测：{prediction.item():.6f}")
    print(f"实际目标：{target.item():.6f}")
    print(f"原顺序损失：{loss.item():.6f}")

    print("\n三个共享参数的梯度：")
    print(f"wx.grad = {wx.grad.item():.8f}")
    print(f"wh.grad = {wh.grad.item():.8f}")
    print(f"b.grad  = {b.grad.item():.8f}")

    # 对照顺序：相同参数、重新从零状态开始，只做前向
    with torch.no_grad():
        reordered_prediction, reordered_records = forward_sequence(
            inputs, reordered_order, wx, wh, b,
            first_six, case="reordered",
        )

    print(f"\n对照顺序预测：{reordered_prediction.item():.6f}")

    trace = pd.DataFrame(original_records + reordered_records)

    print("\n两组状态记录：")
    print(trace.to_string(index=False))

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "rnn_trace.csv"
    trace.to_csv(output_path, index=False)

    print(f"\n记录已保存：{output_path}")

if __name__ == "__main__":
    main()