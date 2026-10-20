# 第五周：MLP 与 PyTorch 训练基础

## 环境与运行

- Python：3.11.9
- PyTorch：2.14.1
- 设备：CPU
- 神经网络数据类型：float32
- 完整依赖版本见 requirements.txt。

在已启用的 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

进入 week05 文件夹后运行：

```bash
python run_step.py
python run_mlp.py
```

如果终端当前位于项目根目录 ml-practice，则运行：

```bash
python week05/run_step.py
python week05/run_mlp.py
```

## 文件职责

| 文件 | 职责 |
|---|---|
| data.py | 准备数据、划分、标准化、转换张量和创建 DataLoader |
| models.py | 创建单层线性模型和可配置隐藏层宽度的 MLP |
| optimizers.py | 创建 SGD 或 Adam 优化器 |
| training.py | 单次更新、逐批逐轮训练、最佳参数记录与恢复、基线拟合 |
| evaluation.py | 无参数更新的评价、预测恢复原始单位、计算误差指标 |
| reporting.py | 保存 CSV 和绘制 PNG，不执行训练 |
| run_step.py | 连接组件，运行实践一 |
| run_mlp.py | 连接组件，运行正式对照实验与宽度检查 |

其他 check_*.py 是分阶段调试文件，hello.py 是环境练习文件。

## 实践一：一次梯度更新

数据：

- x = [[1.0], [2.0]]
- y = [[2.0], [4.0]]
- 模型：Linear(1, 1)，权重和偏置初始化为零。
- 损失：MSE。
- 优化器：SGD，学习率 0.1，无动量、无权重衰减。

一次更新的核心流程：

1. zero_grad() 清理旧梯度。
2. model(x) 计算预测。
3. MSELoss 计算损失。
4. loss.backward() 计算梯度。
5. optimizer.step() 更新参数。

初始 MSE 为 10，权重梯度为 -10，偏置梯度为 -6。
更新后权重为 1，偏置约为 0.6。
重新前向得到预测约为 1.6、2.6，MSE 约为 1.06。

参数更新后，之前保存的预测和损失不会自动重新计算。

结果：results/step_trace.csv。

## 实践二：设置

- 数据：load_diabetes(as_frame=True, scaled=False)，全部 10 个特征。
- 训练/验证/测试数量：265 / 88 / 89。
- 划分：先保留 40%，再将这部分平分；两次 random_state 均为 42。
- 输入标准化：仅在训练集拟合 StandardScaler，验证、测试沿用。
- 目标标准化：使用训练目标均值和 std(ddof=0)。
- 训练目标均值约为 151.0264，标准差约为 77.3308。
- MLP：Linear(10, 16) → ReLU → Linear(16, 1)。
- 最后一层不加 ReLU，允许预测负的标准化目标值。
- batch_size=32，shuffle=True，drop_last=False，num_workers=0。
- 每轮 9 批，最后一批 9 条样本。
- SGD：lr=0.01，momentum=0，weight_decay=0。
- Adam：lr=0.001，weight_decay=0，其余参数默认。
- 两组各训练 200 轮，损失均为标准化目标上的 MSE。
- torch.manual_seed(42) 创建初始模型，通过 deepcopy 创建独立副本。
- 两组各自新建种子为 42 的 DataLoader，保证每轮样本顺序一致。
- 每轮结束，重新评价完整训练集和验证集。
- 验证 MSE 严格降低时保存参数，相同保留较早轮次。
- 训练结束恢复各自最佳参数，再计算原始单位的指标。

基线使用相同数据划分。均值基线预测训练目标均值；
线性回归使用标准化输入与原始目标拟合。

## 实验结果

以下 MLP 指标均来自各自最佳轮次。

| 模型 | 最佳轮次 | 训练 RMSE | 验证 RMSE | 测试 MAE | 测试 RMSE |
|---|---:|---:|---:|---:|---:|
| 均值基线 | — | 77.3308 | 76.1614 | 66.7289 | 76.9057 |
| 线性回归 | — | 54.2849 | 49.1497 | 46.7240 | 56.9703 |
| SGD MLP | 167 | 50.6590 | 50.4665 | — | — |
| Adam MLP | 88 | 50.7106 | 49.8631 | 46.1523 | 55.9007 |

MAE、RMSE 均为原始目标单位。

根据验证 RMSE，在两组 MLP 中选定 Adam；
相同时的选择规则为 SGD。
仅对两个基线和选定 MLP 计算测试指标。
没有合并训练与验证集重训，也没有根据测试结果调整配置。

本次线性回归的验证 RMSE 略低于两个 MLP；
选定 Adam 的测试 RMSE 略低于线性回归。
两组 MLP 的学习率不同，不能把全部差异只归因于优化器算法。

## 输出文件

results/ 中共三份 CSV、两张 PNG：

- step_trace.csv：实践一的三个阶段记录。
- history.csv：两组各 200 轮，共 400 行，MSE 使用标准化目标。
- metrics.csv：四组模型的指标，MAE/RMSE 使用原始单位；
  未选定 SGD 的测试指标留空。
- loss_curves.png：双面板训练/验证 MSE 曲线，标记最佳轮次。
- rmse_comparison.png：四组模型的训练/验证 RMSE。

## 隐藏层宽度修改检查

models.py 中的 hidden_dim 同时控制：

- 第一层 Linear 的输出维数。
- 第二层 Linear 的输入维数。

run_mlp.py 中的 check_hidden_width() 创建 hidden_dim=8 的临时网络。
观察到形状：

```text
(32, 10) → (32, 8) → (32, 8) → (32, 1)
```

该网络只做前向检查，不训练、不追加图表。
正式实验仍使用 hidden_dim=16。

如果只把第一层输出改为 8，第二层输入仍为 16，
两个层的维数不匹配，无法衔接。
最后一层输出维数保持为 1，因此每条样本仍只有一个预测。