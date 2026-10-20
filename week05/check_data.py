from data import prepare_tensor_data, create_train_loader

data = prepare_tensor_data()

for split in ("train", "val", "test"):
    X = data[f"X_{split}_tensor"]
    y = data[f"y_{split}_tensor"]

    print(split, "输入形状：", X.shape)
    print(split, "答案形状：", y.shape)
    print(split, "数据类型：", X.dtype, y.dtype)

loader = create_train_loader(data)

print("\n每批的样本数量：")

batch_sizes = []

for xb, yb in loader:
    assert xb.shape[0] == yb.shape[0]

    batch_sizes.append(xb.shape[0])

print(batch_sizes)
print("一轮的批次数：", len(batch_sizes))
print("一轮的样本总数：", sum(batch_sizes))