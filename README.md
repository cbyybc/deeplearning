# Deep Learning Stock Prediction and Trading Baseline

本项目是“深度学习基础”课程作业中的股票短期收益预测与模拟交易系统。项目基于 A 股日频量价数据，完成了从数据处理、特征工程、标签构造、深度学习模型训练、横截面选股、历史回测、随机策略对比到结果可视化的完整流程。

当前已经完成：

```text
数据处理
→ 特征工程
→ 真实交易标签构造
→ MLP baseline
→ LSTM seq10 e60 baseline
→ LSTM seq20 e60 baseline
→ TopK / Drop / Buffer-Risk 策略
→ 随机策略对比
→ 真实执行假设下的稳健回测
```

本项目默认使用 **Parquet** 格式保存和读取处理后的数据文件。

---

## 1. 项目定位

本项目不是直接预测股票价格，也不是让神经网络直接输出买卖股票列表。

模型的作用是：

```text
输入：某只股票在某个交易日的量价与技术指标特征
输出：该股票未来短期收益的预测分数 pred_score
```

之后由策略模块根据每天所有股票的预测分数排序，生成买入、卖出和继续持有列表。

整体流程为：

```text
特征数据
→ 深度学习模型输出 pred_score
→ 每日横截面排序
→ 交易策略生成目标持仓
→ 历史回测
→ 与随机策略对比
→ 分析模型与策略有效性
```

---

## 2. 当前项目结构

推荐项目目录如下：

```text
Feature Engineering/
│
├── configs/
│   ├── config.json                    # MLP 配置
│   ├── config_lstm.json               # LSTM 初始配置
│   ├── config_lstm_seq10.json         # LSTM seq10 e60 配置
│   └── config_lstm_seq20_e60.json     # LSTM seq20 e60 配置
│
├── src/
│   ├── data_utils.py                  # 数据读取、划分、裁剪、标准化
│   ├── metrics.py                     # IC、RankIC、回测指标
│   ├── model.py                       # MLP 模型
│   ├── train_mlp.py                   # MLP 训练
│   ├── backtest.py                    # 简化回测
│   ├── evaluate.py                    # 验证集评估
│   ├── visualize_results.py           # 可视化
│   │
│   ├── sequence_dataset.py            # LSTM 滑动窗口数据集
│   ├── model_lstm.py                  # LSTM 模型
│   ├── train_lstm.py                  # LSTM 训练
│   └── robust_backtest_lstm.py        # LSTM 稳健回测
│
├── outputs/                           # MLP 输出
├── outputs_lstm_seq10_e60/            # LSTM seq10 e60 输出
├── outputs_lstm_seq20_e60/            # LSTM seq20 e60 输出
│
├── Datasets/                          # 本地数据目录，不上传 GitHub
│
├── Feature Engineering.py             # 特征工程脚本
├── add_realistic_label.py             # 构造 label_oc_1d
├── add_t1_labels.py                   # 可选：构造 T+1 标签
├── robust_backtest_analysis.py        # MLP 稳健回测
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. 数据说明

项目使用 A 股日频量价数据，原始字段主要包括：

```text
ts_code
trade_date
open
high
low
close
pre_close
vol
amount
```

经过特征工程后，当前数据规模大约为：

```text
样本数量：约 747 万
股票数量：3163 只
时间范围：2016-02-01 至 2026-04-28
```

数据质量检查结果：

```text
NaN: 0
inf: 0
平均每个交易日股票数：约 3006
平均每只股票样本数：约 2362
```

处理后的数据默认保存为：

```text
Datasets/processed/all_stock_features_with_oc_label.parquet
```

如果进一步构造 T+1 标签，则保存为：

```text
Datasets/processed/all_stock_features_with_t1_labels.parquet
```

注意：`Datasets/` 不上传 GitHub。

---

## 4. 股票池过滤

特征工程阶段已经进行了基础股票池过滤：

```text
1. 剔除 ST 股票
2. 剔除北交所股票
3. 剔除上市时间过短的股票
4. 剔除缺失严重或无法计算技术指标的样本
```

该处理可以减少异常股票、流动性较差股票和数据不完整股票对模型训练的影响。

---

## 5. 特征工程

当前项目使用的核心特征为 20 个相对特征、技术指标和横截面排名特征。

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

这些特征覆盖了：

```text
1. 横截面相对强弱
2. 短期收益率与动量
3. 均线偏离
4. 波动率
5. 成交量 / 成交额变化
6. RSI 和 MACD 技术指标
7. K 线结构
```

当前 baseline 不直接使用原始价格列作为主要输入，例如：

```text
open
high
low
close
ma5
ma10
ma20
```

原因是不同股票价格尺度差异较大，直接输入可能会干扰模型。当前版本更偏向使用相对特征和标准化后的技术指标。

---

## 6. 标签设计

### 6.1 初始标签：close-to-close

初始实验使用：

```text
label_1d = close[t+1] / close[t] - 1
```

该标签表示从第 t 日收盘到第 t+1 日收盘的收益。

但是该标签存在真实交易问题：

```text
第 t 日收盘后才能生成预测信号，
因此不能再以第 t 日收盘价买入。
```

所以该标签适合做学术上的 close-to-close 预测，但不完全符合真实执行。

### 6.2 当前主标签：open-to-close

当前主实验使用：

```text
label_oc_1d = close[t+1] / open[t+1] - 1
```

含义是：

```text
第 t 日收盘后生成预测信号
第 t+1 日开盘买入
第 t+1 日收盘按市值估值
```

这个标签更符合模拟交易的执行方式。

需要注意：

```text
label_oc_1d 表示次日开盘买入后的当日收盘浮动收益，
不是严格意义上的当天卖出实现收益。
```

在 A 股 T+1 规则下，第二天买入的股票当天不能卖出，但仍然可以按收盘价计算持仓市值变化。因此 `label_oc_1d` 可以用于每日净值 mark-to-market。

### 6.3 T+1 严格标签，可选实验

如果要严格模拟“第 t+1 日买入，第 t+2 日才能卖出”，可以构造：

```text
label_oo_1d = open[t+2] / open[t+1] - 1
```

含义是：

```text
第 t 日收盘后生成信号
第 t+1 日开盘买入
第 t+2 日开盘卖出
```

也可以构造：

```text
label_oc_2d = close[t+2] / open[t+1] - 1
```

当前项目主线使用 `label_oc_1d`，T+1 标签可以作为稳健性补充实验。

---

## 7. 数据划分

为了避免未来信息泄露，所有模型均使用时间划分，而不是随机划分。

默认划分：

```text
训练集：2019-01-01 至 2023-12-31
验证集：2024-01-01 至 2024-12-31
回测集：2025-01-01 至 2025-12-31
```

所有特征裁剪、标签裁剪、标准化参数都只使用训练集计算，再应用到验证集和回测集。

---

## 8. 预处理规则

训练前进行以下处理：

```text
1. 特征裁剪：训练集 1% 和 99% 分位数
2. 标签裁剪：训练集 1% 和 99% 分位数
3. 标准化：只使用训练集均值和标准差
```

验证集、回测集使用训练集计算得到的裁剪阈值与标准化参数。

---

## 9. 已完成模型

当前已经完成：

```text
1. MLP baseline
2. LSTM seq10 e60 baseline
3. LSTM seq20 e60 baseline
```

---

## 10. MLP Baseline

### 10.1 模型说明

MLP 使用单日特征输入：

```text
X.shape = [batch, feature_dim]
```

输出：

```text
pred_score = 未来短期收益预测分数
```

模型结构示例：

```text
20 → 512 → 256 → 128 → 1
```

推荐 A100 配置：

```json
"train": {
  "batch_size": 65536,
  "epochs": 30,
  "lr": 0.002,
  "hidden_dims": [512, 256, 128],
  "dropout": 0.2,
  "loss": "smooth_l1"
}
```

训练命令：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_mlp.py \
  --config configs/config.json
```

稳健回测命令：

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

### 10.2 MLP 主要结果

基于 `label_oc_1d` 的真实执行回测结果：

```text
Realistic IC_mean: 0.02265
ICIR: 0.18847
RankIC_mean: -0.01660
DirectionAcc: 0.49968
```

策略结果：

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
Top10 Full 收益最高，但换手率较高。
Buffer-Risk 收益略低于 Top10 Full，但更稳健、更适合作为低换手主策略。
```

---

## 11. LSTM Baseline

### 11.1 模型说明

LSTM 使用每只股票过去若干天的特征序列作为输入：

```text
X.shape = [batch, seq_len, feature_dim]
```

输出：

```text
pred_score = 未来短期收益预测分数
```

交易含义：

```text
使用第 t 日及以前 seq_len 天特征
预测第 t+1 日开盘到收盘收益 label_oc_1d
```

LSTM 使用相同特征列、相同标签、相同时间划分与相同稳健回测流程，因此可以和 MLP 公平比较。

### 11.2 LSTM seq10 e60

最终 LSTM 主结果采用：

```text
seq_len = 10
epochs = 60
early stopping
best_epoch = 49
```

训练日志显示模型在第 49 轮达到最低验证损失，并在第 57 轮触发 early stopping，说明已经基本收敛。

验证集结果：

```text
IC_mean: 0.03941
ICIR: 0.26401
RankIC_mean: 0.01248
DirectionAcc: 0.52406
valid_loss: 0.000396
best_epoch: 49
```

真实执行回测：

```text
IC_mean: 0.02340
ICIR: 0.26272
RankIC_mean: -0.01122
DirectionAcc: 0.50387
```

策略结果：

| 策略 | 总收益 |
|---|---:|
| LSTM seq10 e60 Top10 Full | 183.77% |
| LSTM seq10 e60 Top10-Drop2 | 172.62% |
| LSTM seq10 e60 Buffer-Risk | 71.53% |
| Random Top10-Drop2 | 51.62% ± 21.62% |
| Random Top10 Full | 40.27% ± 11.13% |

结论：

```text
LSTM seq10 e60 是当前收益表现最强的 LSTM 版本。
它在 Top10 Full 和 Top10-Drop2 策略下明显超过 MLP 和随机策略。
但 Buffer-Risk 表现较弱，说明其信号更偏短周期，需要更积极的换仓机制。
```

### 11.3 LSTM seq20 e60

为了公平比较窗口长度，进一步训练了 seq20 e60。

训练结果：

```text
seq_len = 20
epochs = 60
early stopping at epoch 32
```

这说明 seq20 也已经充分训练，不是训练轮数不足导致的低表现。

策略结果：

| 策略 | 总收益 |
|---|---:|
| LSTM seq20 e60 Top10 Full | 129.27% |
| LSTM seq20 e60 Top10-Drop2 | 97.59% |
| LSTM seq20 e60 Buffer-Risk | 72.72% |
| Random Top10-Drop2 | 54.79% ± 14.90% |
| Random Top10 Full | 34.11% ± 10.56% |

结论：

```text
LSTM seq20 e60 明显超过随机策略，说明长窗口序列模型有效。
但 seq20 e60 仍不如 seq10 e60，说明对于 label_oc_1d 这种短期收益预测任务，较短历史窗口更合适。
```

---

## 12. 模型对比总结

### 12.1 预测指标对比

| 模型 | 窗口 | Realistic IC_mean | ICIR | RankIC_mean |
|---|---:|---:|---:|---:|
| MLP | 单日特征 | 0.02265 | 0.18847 | -0.01660 |
| LSTM | 10 | 0.02340 | 0.26272 | -0.01122 |
| LSTM | 20 | 未记录 | 未记录 | 未记录 |

说明：

```text
MLP 和 LSTM seq10 的 IC 均为正，说明模型具有一定预测能力。
LSTM seq10 e60 的 ICIR 更高，说明预测信号相对更稳定。
RankIC 不高说明全市场完整排序较难，但 TopK 策略仍然可以利用预测分数顶部区域。
```

### 12.2 策略收益对比

| 模型 | Top10 Full | Top10-Drop2 | Buffer-Risk |
|---|---:|---:|---:|
| MLP | 130.33% | 103.33% | 117.08% |
| LSTM seq20 e60 | 129.27% | 97.59% | 72.72% |
| LSTM seq10 e60 | 183.77% | 172.62% | 71.53% |

结论：

```text
1. LSTM seq10 e60 在 Top10 Full 和 Top10-Drop2 策略下表现最强。
2. MLP 在 Buffer-Risk 策略下表现最好。
3. LSTM seq20 e60 虽然有效，但整体不如 seq10 e60。
4. 不同模型的预测信号特性不同，因此最适合的交易策略也不同。
```

---

## 13. 当前推荐策略

当前建议保留两个主策略视角。

### 13.1 收益最优策略

```text
LSTM seq10 e60 + Top10-Drop2
```

理由：

```text
1. 总收益达到 172.62%
2. 明显超过随机 Top10-Drop2
3. 换手率明显低于 Top10 Full
4. 更适合短期序列信号
```

虽然 LSTM seq10 e60 + Top10 Full 收益更高，但换手率也更高，真实交易中更容易受到手续费、滑点和成交风险影响。

### 13.2 稳健低换手策略

```text
MLP + Buffer-Risk
```

理由：

```text
1. 总收益达到 117.08%
2. 低于 LSTM seq10 e60 的最高收益，但更稳健
3. 风险控制逻辑更清晰
4. 更适合作为保守策略和可解释性对照
```

---

## 14. 运行命令汇总

### 14.1 生成 open-to-close 标签

```bash
python add_realistic_label.py
```

### 14.2 训练 MLP

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_mlp.py \
  --config configs/config.json
```

### 14.3 MLP 稳健回测

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

### 14.4 训练 LSTM seq10 e60

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_lstm.py \
  --config configs/config_lstm_seq10.json
```

### 14.5 LSTM seq10 e60 稳健回测

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/robust_backtest_lstm.py \
  --config configs/config_lstm_seq10.json \
  --model_path outputs_lstm_seq10_e60/models/best_lstm.pt \
  --out_dir outputs_lstm_seq10_e60/robust_backtest_lstm \
  --top_k 10 \
  --drop_k 2 \
  --fee_rate 0.0003 \
  --random_runs 20
```

### 14.6 训练 LSTM seq20 e60

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/train_lstm.py \
  --config configs/config_lstm_seq20_e60.json
```

### 14.7 LSTM seq20 e60 稳健回测

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python src/robust_backtest_lstm.py \
  --config configs/config_lstm_seq20_e60.json \
  --model_path outputs_lstm_seq20_e60/models/best_lstm.pt \
  --out_dir outputs_lstm_seq20_e60/robust_backtest_lstm \
  --top_k 10 \
  --drop_k 2 \
  --fee_rate 0.0003 \
  --random_runs 20
```

---

## 15. 可视化输出

建议报告中使用以下图：

```text
1. MLP loss_curve.png
2. MLP monthly_ic.png
3. MLP strategy_nav_comparison.png
4. MLP strategy_drawdown_comparison.png
5. LSTM seq10 e60 loss_curve_lstm.png
6. LSTM seq10 e60 monthly_ic_lstm.png
7. LSTM seq10 e60 strategy_nav_comparison_lstm.png
8. LSTM seq10 e60 strategy_drawdown_comparison_lstm.png
9. LSTM seq20 e60 strategy_nav_comparison_lstm.png
```

重点结果文件：

```text
strategy_nav_comparison.png
strategy_drawdown_comparison.png
strategy_metrics_summary.csv
realistic_prediction_metrics.json
monthly_ic.png
```

---

## 16. GitHub 上传说明

本项目不上传大型数据文件和模型权重。

`.gitignore` 应忽略：

```text
Datasets/
*.parquet
*.pkl
*.pickle
*.pt
*.pth
outputs/models/
outputs_lstm*/models/
outputs*/predictions/
```

建议上传：

```text
源代码
配置文件
README.md
requirements.txt
关键结果图
小型 JSON 指标
strategy_metrics_summary.csv
```

不建议上传：

```text
原始数据
处理后的 Parquet 数据
模型权重
大型预测 CSV
大型持仓记录
大型交易记录
```

---

## 17. 当前结论

目前已经完成的核心成果：

```text
1. 完成全市场 A 股数据特征工程
2. 构造了更符合真实执行的 label_oc_1d
3. 完成 MLP baseline
4. 完成 LSTM seq10 e60 baseline
5. 完成 LSTM seq20 e60 baseline
6. 完成随机策略对比
7. 完成真实执行假设下的稳健回测
8. 完成多模型、多策略对比
```

当前最重要实验发现：

```text
1. LSTM seq10 e60 是当前收益最强的短线模型。
2. MLP + Buffer-Risk 是更稳健、低换手、可解释的策略。
3. seq20 e60 充分训练后仍不如 seq10 e60，说明短窗口更适合 label_oc_1d。
4. RankIC 不高说明全市场完整排序较难，但 TopK 策略仍然有效。
5. 不同模型适合不同交易策略，不能只看单一 IC 指标。
6. 标签定义必须和真实交易执行方式保持一致。
```

---

## 18. 下一步计划

后续建议继续完成：

```text
1. DLinear baseline
2. MLP / LSTM / DLinear 统一对比
3. 更严格的 T+1 标签实验 label_oo_1d
4. 更完整的交易约束：涨跌停、停牌、滑点
5. 最终实验报告
6. 模拟交易每日预测脚本 predict_daily.py
```

推荐下一步模型：

```text
DLinear
```

原因：

```text
DLinear 是 LTSF 领域常用强 baseline。
实现简单。
可以与 MLP 和 LSTM 形成清晰模型对比。
即使结果不超过 LSTM，也能作为有效实验分析。
```

---

## 19. 报告可用总结

可以在报告中这样描述当前结果：

```text
本文首先构建了 MLP baseline，用单日量价技术指标预测未来短期收益分数。随后，为了检验序列建模是否能够提升预测效果，进一步实现了 LSTM 模型，并分别测试了 10 日和 20 日历史窗口。实验结果表明，LSTM seq10 在充分训练后于 Top10 Full 和 Top10-Drop2 策略下取得最高收益，而 MLP 在 Buffer-Risk 策略下表现更稳健。这说明不同模型产生的预测信号具有不同特性，需要与合适的交易策略结合。

同时，本文发现初始 close-to-close 标签与真实交易执行方式不完全一致，因此构造了 open-to-close 标签 label_oc_1d，使训练目标更接近“收盘后生成信号、次日开盘买入”的真实流程。与随机选股策略相比，MLP 和 LSTM 策略均取得更高收益，说明模型预测分数具有一定实际选股价值。
```

---

## 20. 项目当前状态

```text
MLP baseline: 已完成
LSTM seq10 e60: 已完成
LSTM seq20 e60: 已完成
随机策略对比: 已完成
稳健回测: 已完成
GitHub 上传: 已完成主体内容
DLinear: 待完成
最终报告: 待完成
```
