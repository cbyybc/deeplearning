# MLP Baseline：基于深度学习的股票短期收益预测与回测

本项目是“基于深度学习的股票趋势预测与模拟交易”大作业中的 **MLP baseline**。  
项目目标不是直接预测股票价格，也不是让神经网络直接输出买卖股票列表，而是构建一个完整、可复现的基础流程：

```text
特征数据
→ MLP 预测每只股票的未来收益分数
→ 每日横截面排序
→ TopK / Buffer-Risk 策略生成持仓
→ 历史回测
→ 与随机策略对比
→ 为后续 LSTM / DLinear / PatchTST 做 baseline
```

当前版本使用 **Parquet** 作为默认数据格式。

---

## 1. 项目核心思想

MLP 模型的输入是每只股票在某个交易日的量价与技术指标特征，输出是一个预测分数 `pred_score`。

该分数不是严格意义上的“明天涨幅”，而是用于横截面排序的股票未来收益评分：

```text
pred_score 高 → 模型认为该股票未来表现相对更好
pred_score 低 → 模型认为该股票未来表现相对更差
```

最终买入、卖出、继续持有由策略模块完成，而不是 MLP 直接输出。

---

## 2. 当前推荐训练目标

本项目最终推荐使用：

```text
label_oc_1d = close[t+1] / open[t+1] - 1
```

含义是：

```text
第 t 日收盘后生成预测信号；
第 t+1 日开盘买入；
第 t+1 日收盘计算收益。
```

相比原始的：

```text
label_1d = close[t+1] / close[t] - 1
```

`label_oc_1d` 更符合实际模拟交易执行逻辑，因为我们不可能在第 t 日收盘后再以第 t 日收盘价买入。

---

## 3. 数据文件要求

默认数据文件为：

```text
Datasets/processed/all_stock_features_with_oc_label.parquet
```

该文件应至少包含以下字段：

```text
ts_code
trade_date
open
high
low
close
vol
amount
label_oc_1d
```

以及已经构造好的特征字段。

如果你尚未生成 `label_oc_1d`，需要先运行：

```bash
python add_realistic_label.py
```

生成：

```text
Datasets/processed/all_stock_features_with_oc_label.parquet
```

---

## 4. 当前使用的特征

本 baseline 不直接使用原始价格作为主要输入，而优先使用相对特征、技术指标和横截面排名特征。

默认特征列如下：

```python
feature_cols = [
    "close_rank",
    "vol_rank",
    "amount_rank",

    "amplitude",
    "body_ratio",

    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",

    "rel_ma5",
    "rel_ma10",
    "rel_ma20",

    "rsi_14",

    "macd_dif",
    "macd_dea",
    "macd_bar",

    "vol_std_10",
    "vol_std_20",

    "vol_chg",
    "amount_chg",
]
```

禁止作为模型输入的字段包括：

```text
ts_code
trade_date
label_1d
label_2d
label_5d
label_10d
label_oc_1d
label_direction_1d
next_close
next_open
next_close_real
```

这些字段要么是标识符，要么包含未来信息，不能进入模型输入。

---

## 5. 数据划分

为了避免未来信息泄露，数据集必须按时间划分，不能随机划分。

当前默认划分：

```text
训练集：2019-01-01 至 2023-12-31
验证集：2024-01-01 至 2024-12-31
回测集：2025-01-01 至 2025-12-31
```

对应配置在：

```text
configs/config.json
```

---

## 6. 预处理规则

训练前会自动执行以下处理：

### 6.1 特征极端值裁剪

对每个特征使用训练集的 1% 和 99% 分位数进行裁剪：

```text
feature = clip(feature, train_1_percentile, train_99_percentile)
```

验证集和回测集使用训练集计算得到的上下界。

### 6.2 标签极端值裁剪

对标签 `label_oc_1d` 使用训练集分位数裁剪：

```text
label_oc_1d = clip(label_oc_1d, train_1_percentile, train_99_percentile)
```

### 6.3 标准化

只使用训练集均值和标准差：

```text
x = (x - train_mean) / train_std
```

验证集和回测集使用同一组训练集统计量。

---

## 7. 模型结构

当前 MLP 结构由 `configs/config.json` 控制。

推荐 A100 版本：

```json
"train": {
  "seed": 42,
  "batch_size": 65536,
  "epochs": 30,
  "lr": 0.002,
  "weight_decay": 0.0001,
  "hidden_dims": [512, 256, 128],
  "dropout": 0.2,
  "early_stop_patience": 5,
  "loss": "smooth_l1"
}
```

模型结构：

```text
input_dim = 20
↓
Linear(20, 512)
BatchNorm
ReLU
Dropout
↓
Linear(512, 256)
BatchNorm
ReLU
Dropout
↓
Linear(256, 128)
BatchNorm
ReLU
Dropout
↓
Linear(128, 1)
↓
pred_score
```

损失函数默认使用 `SmoothL1Loss`，比 MSE 对异常收益更稳健。

---

## 8. 环境安装

```bash
pip install -r requirements.txt
```

推荐依赖：

```text
numpy
pandas
pyarrow
torch
scikit-learn
matplotlib
tqdm
```

如果运行时出现 protobuf / onnx 相关报错，可临时使用：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_mlp.py --config configs/config.json
```

或者安装兼容版本：

```bash
pip install "protobuf==3.20.3"
```

---

## 9. 配置文件

配置文件路径：

```text
configs/config.json
```

推荐内容示例：

```json
{
  "data_path": "Datasets/processed/all_stock_features_with_oc_label.parquet",
  "output_dir": "outputs",
  "label_col": "label_oc_1d",

  "feature_cols": [
    "close_rank",
    "vol_rank",
    "amount_rank",
    "amplitude",
    "body_ratio",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "rel_ma5",
    "rel_ma10",
    "rel_ma20",
    "rsi_14",
    "macd_dif",
    "macd_dea",
    "macd_bar",
    "vol_std_10",
    "vol_std_20",
    "vol_chg",
    "amount_chg"
  ],

  "split": {
    "train_start": 20190101,
    "train_end": 20231231,
    "valid_start": 20240101,
    "valid_end": 20241231,
    "backtest_start": 20250101,
    "backtest_end": 20251231
  },

  "preprocess": {
    "feature_clip_lower": 0.01,
    "feature_clip_upper": 0.99,
    "label_clip_lower": 0.01,
    "label_clip_upper": 0.99
  },

  "train": {
    "seed": 42,
    "batch_size": 65536,
    "epochs": 30,
    "lr": 0.002,
    "weight_decay": 0.0001,
    "hidden_dims": [512, 256, 128],
    "dropout": 0.2,
    "early_stop_patience": 5,
    "loss": "smooth_l1"
  },

  "backtest": {
    "top_k": 10,
    "drop_k": 2,
    "initial_nav": 1.0,
    "fee_rate": 0.0003
  }
}
```

---

## 10. 训练 MLP

在项目根目录运行：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_mlp.py --config configs/config.json
```

训练完成后会生成：

```text
outputs/models/best_mlp.pt
outputs/predictions/valid_predictions.csv
outputs/metrics_valid.json
outputs/training_history.csv
outputs/figures/loss_curve.png
```

---

## 11. 验证集指标

训练脚本会自动输出验证集指标，包括：

```text
Valid Loss
IC_mean
ICIR
RankIC_mean
RankICIR
DirectionAcc
num_days
num_samples
```

指标含义：

| 指标 | 含义 |
|---|---|
| IC | 每日横截面预测分数与真实收益的 Pearson 相关 |
| ICIR | IC 均值 / IC 标准差 |
| RankIC | 每日横截面预测排序与真实收益排序的 Spearman 相关 |
| DirectionAcc | 预测方向与真实方向一致比例 |
| Valid Loss | 验证集回归损失 |

---

## 12. 原始简化回测

运行：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/backtest.py --config configs/config.json
```

输出：

```text
outputs/backtest/backtest_predictions.csv
outputs/backtest/nav_curve.csv
outputs/backtest/trade_records.csv
outputs/backtest/position_records.csv
outputs/backtest/backtest_metrics.json
outputs/figures/backtest_nav.png
outputs/figures/backtest_drawdown.png
```

注意：该回测主要用于快速验证模型流程，真实可信度低于 `robust_backtest_analysis.py`。

---

## 13. 稳健回测与随机策略对比

推荐最终报告使用这个回测结果。

运行：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python robust_backtest_analysis.py \
  --config configs/config.json \
  --model_path outputs/models/best_mlp.pt \
  --out_dir outputs/robust_backtest_oc_label \
  --top_k 10 \
  --drop_k 2 \
  --fee_rate 0.0003 \
  --random_runs 20
```

该脚本会进行：

```text
1. next-day open-to-close realistic 回测
2. MLP Top10 全换策略
3. MLP Top10-Drop2 策略
4. MLP Buffer-Risk 策略
5. Random Top10 全换策略
6. Random Top10-Drop2 策略
7. 月度 IC 分析
8. 策略净值曲线对比
9. 策略回撤曲线对比
```

输出目录：

```text
outputs/robust_backtest_oc_label/
```

核心输出文件：

```text
realistic_signal_predictions.csv
realistic_prediction_metrics.json
monthly_ic.csv
monthly_ic.png
strategy_metrics_all_runs.csv
strategy_metrics_summary.csv
strategy_nav_comparison.png
strategy_drawdown_comparison.png
```

---

## 14. 当前 MLP baseline 结果

基于 `label_oc_1d` 重新训练后，真实执行回测结果如下：

### 14.1 预测指标

```text
IC_mean: 0.02265
IC_std: 0.12019
ICIR: 0.18847
RankIC_mean: -0.01660
RankICIR: -0.10625
DirectionAcc: 0.49968
num_days: 242
num_samples: 762278
```

解释：

```text
模型对 next-day open-to-close 收益具有一定线性预测能力；
RankIC 为负，说明模型并非对全市场完整排序都稳定有效；
但 Top 端选股策略表现较好，说明模型可能主要在高分股票区域具有选股价值。
```

### 14.2 策略结果

| 策略 | 总收益 |
|---|---:|
| MLP Top10 Full | 130.33% |
| MLP Buffer-Risk | 117.08% |
| MLP Top10-Drop2 | 103.33% |
| Random Top10-Drop2 | 60.80% ± 13.03% |
| Random Top10 Full | 50.13% ± 17.51% |

结论：

```text
MLP 策略明显超过随机策略，说明模型预测分数具有实际选股价值。
Top10 Full 收益最高，但换手率较高；
Buffer-Risk 收益略低，但更稳定、更适合作为主策略。
```

---

## 15. 推荐最终主策略

虽然 `Top10 Full` 当前收益最高，但它换手率过高，真实交易中更容易受到手续费、滑点和成交价格影响。

推荐报告主策略使用：

```text
MLP Buffer-Risk Strategy
```

策略逻辑：

```text
1. 每日使用 MLP 对所有股票输出 pred_score
2. 只从预测排名前 20 的股票中选择买入候选
3. 当前持仓如果仍在预测排名前 50，则继续持有
4. 跌出前 50 的持仓进入卖出候选
5. 每天最多替换 3 只股票
6. 买入时过滤短期涨跌过大、高波动、低流动性股票
7. 等权持仓 10 只股票
```

该策略兼顾：

```text
收益
换手率
风险控制
可解释性
模拟交易可执行性
```

---

## 16. 可视化

如果需要生成更完整的可视化图，可运行：

```bash
python src/visualize_results.py --output_dir outputs
```

或者直接查看稳健回测输出：

```text
outputs/robust_backtest_oc_label/monthly_ic.png
outputs/robust_backtest_oc_label/strategy_nav_comparison.png
outputs/robust_backtest_oc_label/strategy_drawdown_comparison.png
```

建议报告中至少放入：

```text
1. 训练 / 验证 loss 曲线
2. 月度 IC 图
3. 策略净值对比图
4. 策略回撤对比图
5. 策略指标对比表
```

---

## 17. 推荐报告表述

可以在报告中这样描述本 baseline：

```text
本文首先构建 MLP baseline，用于验证量价特征对股票短期收益预测的有效性。模型并不直接输出交易指令，而是对每个交易日的每只股票输出未来收益预测分数。随后，策略模块根据预测分数进行横截面排序，并结合 TopK、持仓缓冲区和风险过滤规则生成每日目标持仓。

初始实验使用 close-to-close 收益作为标签，但在更真实的次日开盘执行回测中效果下降。因此，本文进一步构造 open-to-close 标签 label_oc_1d，使训练目标与实际交易执行方式保持一致。实验结果显示，基于该标签训练的 MLP 在真实执行回测中显著超过随机选股策略，说明模型输出分数具有一定实际选股价值。
```

---

## 18. 常见问题

### 18.1 MLP 是预测股价吗？

不是。MLP 预测的是未来收益分数，不是明天股价。

### 18.2 MLP 会直接输出买哪些股票吗？

不会。MLP 只输出每只股票的 `pred_score`，买卖决策由策略模块完成。

### 18.3 为什么要用 `label_oc_1d`？

因为真实模拟交易中，我们通常是在第 t 日收盘后生成信号，第 t+1 日开盘后才能买入，所以 `open[t+1]` 到 `close[t+1]` 的收益更接近实际可执行收益。

### 18.4 为什么 Random 策略也赚钱？

说明 2025 年回测区间股票池整体环境较好。  
因此必须用 MLP 策略和 Random 策略对比，而不能只看绝对收益。

### 18.5 为什么 RankIC 是负的，但策略收益很好？

因为策略只关心预测分数最高的一小部分股票，而 RankIC 衡量全市场完整排序。模型可能在 Top 端选股有效，但对全市场中间排序不稳定。

---

## 19. 下一步工作

MLP baseline 完成后，建议继续：

```text
1. 实现 LSTM / GRU，对比序列建模是否优于 MLP
2. 实现 DLinear，作为 LTSF baseline
3. 对比不同策略：Top10 Full、Top10-Drop2、Buffer-Risk
4. 加入更严格交易约束：涨跌停、停牌、滑点
5. 准备每日模拟交易脚本 predict_daily.py
```

---

## 20. 项目定位

本项目是完整量化深度学习交易系统的第一版 baseline，主要作用是：

```text
验证数据处理是否正确
验证标签定义是否合理
验证模型能否产生有效选股分数
验证 IC / 回测 / 随机对比流程
为后续 LSTM、DLinear、PatchTST 提供对照组
```

不建议直接把 MLP 结果当作最终最优模型，但它已经可以作为报告中的重要 baseline。
