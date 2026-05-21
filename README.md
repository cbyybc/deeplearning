# Deep Learning Stock Prediction and Trading Baseline

本项目是“深度学习基础”课程大作业中的股票短期收益预测与模拟交易系统。项目基于 A 股日频量价数据，完成了从数据处理、特征工程、标签构造、模型训练、横截面选股、历史回测、随机策略对比到 T+1 真实交易口径回测的完整流程。

截至 2026-05-21，项目已经完成两套标签实验、严格 T+1 回测、市场基准、交易成本敏感性分析、每日候选预测脚本、比赛交易计划脚本和图形化交易助手：

```text
1. label_oc_1d
   close[t+1] / open[t+1] - 1
   第 t 日收盘生成信号，第 t+1 日开盘买入，第 t+1 日收盘估值。

2. label_oo_1d
   open[t+2] / open[t+1] - 1
   第 t 日收盘生成信号，第 t+1 日开盘买入，第 t+2 日开盘卖出。
   这是更严格的 A 股 T+1 open-to-open 实验口径。
```

已完成模型：

```text
MLP baseline
LSTM seq10 e60
LSTM seq20 e60
DLinear seq10 e60
DLinear seq20 e60
```

## 项目流程

```text
原始 A 股日频数据
-> 股票池过滤
-> 特征工程
-> label_oc_1d / label_oo_1d 标签构造
-> 时间序列划分
-> 训练集预处理参数拟合
-> MLP / LSTM / DLinear 模型训练
-> 验证集 IC / RankIC / DirectionAcc 评估
-> 2025 年真实执行口径回测
-> Top10 / Drop2 / Buffer-Risk 策略对比
-> 随机策略基准对比
-> 市场指数基准与交易成本敏感性分析
-> 最新盘后数据生成每日候选股票
-> 当前持仓生成比赛买入 / 卖出 / 持有计划
-> 同花顺模拟盘人工执行与结果记录
```

## 数据与划分

项目使用 A 股日频量价数据，主要字段包括：

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

特征工程后的数据规模约为：

```text
样本数量：约 747 万
股票数量：3163 只
时间范围：2016-02-01 至 2026-04-28
NaN: 0
inf: 0
平均每个交易日股票数：约 3006
平均每只股票样本数：约 2362
```

默认时间划分：

```text
训练集：2019-01-01 至 2023-12-31
验证集：2024-01-01 至 2024-12-31
回测集：2025-01-01 至 2025-12-31
```

所有特征裁剪、标签裁剪、标准化参数均只在训练集上拟合，再应用到验证集和回测集，避免未来信息泄露。

## 特征工程

当前模型使用 20 个核心特征：

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

这些特征覆盖横截面排名、短期动量、均线偏离、波动率、量价变化、RSI、MACD 和 K 线结构。

## 新增 T+1 回测代码

新增脚本：

```text
src/robust_backtest_t1_all_models.py
src/benchmark_index_local.py
src/fee_sensitivity.py
src/predict_daily.py
src/make_trade_plan.py
src/trading_dashboard.py
```

`src/robust_backtest_t1_all_models.py` 统一支持 MLP、LSTM 和 DLinear 的严格 T+1 open-to-open 回测：

```text
signal_date = t
buy_date    = t+1 open
sell_date   = t+2 open
realized_ret = open[t+2] / open[t+1] - 1
```

脚本会输出：

```text
t1_prediction_metrics.json
strategy_metrics_summary_t1.csv
monthly_ic_t1.csv
strategy_nav_comparison_t1.png
strategy_drawdown_comparison_t1.png
```

`src/benchmark_index_local.py` 读取本地指数 CSV，输出 2025 年市场指数基准净值与回撤；`src/fee_sensitivity.py` 基于策略 NAV 中的 `portfolio_ret` 和 `turnover` 重算不同费率下的净值，用于评估手续费和滑点近似冲击。

`src/predict_daily.py` 面向比赛阶段的每日盘后预测：读取最新股票日频数据，复用训练期 scaler 和 LSTM checkpoint，按 `signal_date` 生成全股票池候选分数文件。默认输出：

```text
outputs_daily/daily_candidates_<signal_date>.csv
outputs_daily/latest_candidates.csv
outputs_daily/trade_plan_<signal_date>_for_<trade_date>.csv
outputs_daily/latest_trade_plan.csv
```

`src/make_trade_plan.py` 面向人工下单前的交易计划整理：读取最新候选分数和 `data/current_positions.csv`，按 TopK / DropK 策略输出每日买入、卖出、继续持有清单。默认输出：

```text
outputs_trade_plan/buy_list_<trade_date>.csv
outputs_trade_plan/sell_list_<trade_date>.csv
outputs_trade_plan/hold_list_<trade_date>.csv
outputs_trade_plan/trade_plan_<trade_date>.csv
outputs_trade_plan/latest_buy_list.csv
outputs_trade_plan/latest_sell_list.csv
outputs_trade_plan/latest_hold_list.csv
outputs_trade_plan/latest_trade_plan.csv
```

这两个脚本补齐了作业要求中的比赛落地链路：最新日期可以生成预测结果，交易计划能记录买入、卖出和持有动作，策略参数默认采用 `Top10-Drop2`，与作业建议的 `n = 5-30`、`k = 1-5` 区间一致。

`src/trading_dashboard.py` 将每日预测和交易计划脚本封装成 Streamlit 图形界面，适合比赛期间重复操作。界面支持上传或选择最新行情数据、编辑真实持仓、运行预测、生成买卖持有清单、下载 CSV，并增加比赛执行预检：

```text
TopK / DropK 合法性检查
候选信号日期与交易日期的 T+1 提示
预测脚本、交易脚本、checkpoint、scaler 就绪状态
北交所与 ST 候选风险提示
清空交易计划输出前的确认
```

## 输出目录

```text
outputs/
  MLP label_oc_1d baseline

outputs_mlp_oo/
  MLP label_oo_1d T+1 baseline

outputs_lstm_seq10_e60/
  LSTM seq10 label_oc_1d

outputs_lstm_seq10_oo_e60/
  LSTM seq10 label_oo_1d

outputs_lstm_seq20_e60/
  LSTM seq20 label_oc_1d

outputs_lstm_seq20_oo_e60/
  LSTM seq20 label_oo_1d

outputs_dlinear_seq10_e60/
  DLinear seq10 label_oc_1d

outputs_dlinear_seq10_oo_e60/
  DLinear seq10 label_oo_1d

outputs_dlinear_seq20_e60/
  DLinear seq20 label_oc_1d

outputs_dlinear_seq20_oo_e60/
  DLinear seq20 label_oo_1d

outputs_benchmark/index/
  2025 年市场指数基准结果

outputs_benchmark/fee_sensitivity/
  严格 T+1 候选策略交易成本敏感性结果

outputs_daily/
  每日盘后候选股票与最新交易计划

outputs_trade_plan/
  比赛买入、卖出、持有拆分清单
```

## 本地权重运行图形化程序教程

图形化交易助手可以完全在本地运行，不需要服务器。只要本机具备模型推理所需文件，界面会在本地调用预测脚本和交易计划脚本。

### 1. 本地运行需要什么

| 类型 | 文件或目录 | 作用 | 是否随代码仓库提供 |
|---|---|---|---|
| 代码 | `src/trading_dashboard.py` | Streamlit 图形界面 | 需要上传到 GitHub |
| 代码 | `src/predict_daily.py` | 使用权重生成每日候选分数 | GitHub 已有 |
| 代码 | `src/make_trade_plan.py` | 生成买入、卖出、持有清单 | GitHub 已有 |
| 依赖 | `requirements.txt` | 安装 Python 依赖，需包含 `streamlit` | 需要上传最新版 |
| 权重 | `best_model.pt` | LSTM 模型参数 | 通常单独下载，不建议直接放 GitHub |
| 预处理参数 | `scaler.json` | 训练期特征标准化参数 | 必须与权重配套提供 |
| 行情数据 | 最新股票 `csv` / `parquet` 文件 | 生成信号日特征序列 | 由使用者本地准备 |
| 持仓文件 | `data/current_positions.csv` | 记录同花顺真实持仓 | 可由界面创建 |

注意：

```text
1. 只有 best_model.pt 不足以复现每日预测，必须同时有配套 scaler.json。
2. checkpoint、scaler、特征列顺序和预测脚本必须来自同一套训练流程。
3. 数据和模型权重一般不随课程代码仓库上传，下载代码后需要手动放到本地指定位置。
```

### 2. 推荐本地目录结构

为了让 dashboard 默认路径直接可用，推荐把下载的权重和 scaler 放到以下位置：

```text
deeplearning/
  requirements.txt
  README.md
  src/
    trading_dashboard.py
    predict_daily.py
    make_trade_plan.py
  outputs_lstm_seq10_oo_e60/
    best_model.pt
    scaler.json
  data/
    current_positions.csv
  local_market_data/
    000001.SZ.csv
    000002.SZ.csv
    ...
```

`local_market_data/` 也可以换成你自己的最新行情目录。预测脚本会读取该目录中的 `csv` 或 `parquet` 文件。

### 3. 下载代码并准备环境

在本机安装 Python 后，进入项目目录安装依赖：

```powershell
pip install -r requirements.txt
```

如果使用虚拟环境，可以先执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. 放入下载的权重和 scaler

将下载得到的模型文件放到：

```text
outputs_lstm_seq10_oo_e60/best_model.pt
```

将与该权重配套的标准化文件放到：

```text
outputs_lstm_seq10_oo_e60/scaler.json
```

如果权重和 scaler 放在其他目录，也可以启动 dashboard 后在左侧栏手动填写完整路径。

### 5. 启动图形化程序

如果 dashboard 位于仓库 `src/` 目录：

```powershell
streamlit run src/trading_dashboard.py
```

如果 dashboard 当前位于项目根目录：

```powershell
streamlit run trading_dashboard.py
```

Streamlit 会在本机启动网页界面。浏览器页面只是本地操作面板，模型推理、CSV 读写和交易计划生成都在本机完成。

### 6. 在界面中完成一次预测和交易计划

1. 在侧边栏确认 `项目根目录`、`checkpoint`、`scaler.json`、`TopK=10`、`DropK=2`、`signal_date` 和计划交易日期。
2. 在“上传/选择数据”页选择最新行情目录，或上传本次使用的 `csv`、`parquet`、`zip` 数据。
3. 在“当前持仓”页创建或更新 `data/current_positions.csv`。第一次建仓时可先使用空持仓。
4. 在“生成预测”页运行每日预测，生成 `outputs_daily/latest_candidates.csv`。
5. 在“生成交易计划”页生成交易计划。
6. 在“查看与下载”页下载：

```text
latest_buy_list.csv
latest_sell_list.csv
latest_hold_list.csv
latest_trade_plan.csv
```

7. 在同花顺模拟盘按清单人工交易，成交后回到界面更新真实持仓。

### 7. 常见问题

| 问题 | 优先检查 |
|---|---|
| `No module named streamlit` | 重新执行 `pip install -r requirements.txt` |
| 找不到 `best_model.pt` | 检查 checkpoint 路径是否指向下载权重 |
| 找不到 `scaler.json` | 确认下载了与权重配套的 scaler |
| `No valid sequence generated` | 检查行情目录是否包含信号日数据，且每只股票有足够历史窗口 |
| 交易日期不对 | 显式填写 `signal_date` 和 `trade_date`，周末/节假日不要依赖自然日加一 |
| 候选中出现不合规股票 | 再次核对北交所与 ST 过滤结果 |

## A. label_oc_1d 实验结果

### MLP baseline

config: `configs/config.json`

```text
seq_len = N/A
hidden_dims = [512, 256, 128]
epochs = 30
batch_size = 65536
lr = 0.002
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 5
```

验证集结果：

```text
IC_mean: 0.03484
ICIR: 0.20492
RankIC_mean: 0.01051
RankICIR: 0.05309
DirectionAcc: 0.50532
valid_loss: 0.00039402
best_epoch: 16
num_days: 242
num_samples: 762816
```

真实执行回测：

```text
IC_mean: 0.02265
ICIR: 0.18847
RankIC_mean: -0.01660
RankICIR: -0.10625
DirectionAcc: 0.49968
num_days: 242
num_samples: 762278
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| MLP Top10 Full | 130.33% | 138.41% | 2.89 | -12.86% | 56.61% | 1.6826 |
| MLP Buffer-Risk | 117.08% | 124.14% | 2.55 | -9.30% | 60.74% | 0.5992 |
| MLP Top10-Drop2 | 103.33% | 109.38% | 2.59 | -13.35% | 55.79% | 0.4025 |
| Random Top10-Drop2 | 60.80% +- 13.03% | 64.01% +- 13.86% | 2.29 +- 0.41 | -9.24% +- 1.89% | 57.46% +- 1.68% | 0.4025 |
| Random Top10 Full | 50.13% +- 17.51% | 52.71% +- 18.54% | 1.94 +- 0.62 | -9.95% +- 1.80% | 57.42% +- 2.71% | 1.9893 |

### LSTM seq10 e60

config: `configs/config_lstm_seq10.json`

```text
seq_len = 10
hidden_size = 128
num_layers = 2
bidirectional = false
epochs = 60
batch_size = 32768
lr = 0.001
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.03941
ICIR: 0.26401
RankIC_mean: 0.01248
RankICIR: 0.06733
DirectionAcc: 0.52406
valid_loss: 0.00039613
best_epoch: 49
num_days: 233
num_samples: 734349
```

真实执行回测：

```text
IC_mean: 0.02340
ICIR: 0.26272
RankIC_mean: -0.01122
RankICIR: -0.09216
DirectionAcc: 0.50387
num_days: 233
num_samples: 733855
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM seq10 Top10 Full | 183.77% | 208.96% | 3.97 | -13.89% | 62.23% | 1.6489 |
| LSTM seq10 Top10-Drop2 | 172.62% | 195.85% | 3.85 | -11.36% | 62.66% | 0.4026 |
| LSTM seq10 Buffer-Risk | 71.53% | 79.25% | 2.39 | -11.35% | 60.52% | 0.5991 |
| Random Top10-Drop2 | 51.62% +- 21.62% | 56.99% +- 24.24% | 2.31 +- 0.74 | -9.24% +- 2.01% | 58.20% +- 2.41% | 0.4026 |
| Random Top10 Full | 40.27% +- 11.13% | 44.23% +- 12.37% | 1.96 +- 0.44 | -9.78% +- 2.06% | 56.57% +- 2.47% | 1.9882 |

### LSTM seq20 e60

config: `configs/config_lstm_seq20_e60.json`

```text
seq_len = 20
hidden_size = 128
num_layers = 2
bidirectional = false
epochs = 60
batch_size = 16384
lr = 0.001
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.03036
ICIR: 0.18979
RankIC_mean: 0.00035
RankICIR: 0.00175
DirectionAcc: 0.51802
valid_loss: 0.00039405
best_epoch: 24
num_days: 223
num_samples: 702719
```

真实执行回测：

```text
IC_mean: 0.02179
ICIR: 0.24889
RankIC_mean: -0.01131
RankICIR: -0.09784
DirectionAcc: 0.50321
num_days: 223
num_samples: 702280
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM seq20 Top10 Full | 129.27% | 155.39% | 3.26 | -15.27% | 58.30% | 1.5946 |
| LSTM seq20 Top10-Drop2 | 97.59% | 115.89% | 2.69 | -12.94% | 55.16% | 0.4027 |
| LSTM seq20 Buffer-Risk | 72.72% | 85.45% | 2.52 | -11.20% | 56.95% | 0.6018 |
| Random Top10-Drop2 | 54.79% +- 14.90% | 63.94% +- 17.85% | 2.65 +- 0.57 | -9.29% +- 1.90% | 58.86% +- 2.35% | 0.4027 |
| Random Top10 Full | 34.11% +- 10.56% | 39.39% +- 12.42% | 1.84 +- 0.45 | -10.48% +- 1.89% | 56.46% +- 2.37% | 1.9887 |

### DLinear seq10 e60

config: `configs/config_dlinear_seq10_e60.json`

```text
seq_len = 10
moving_avg = 3
epochs = 60
batch_size = 65536
lr = 0.001
weight_decay = 0.0001
dropout = 0.1
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.01889
ICIR: 0.13304
RankIC_mean: -0.01195
RankICIR: -0.06131
DirectionAcc: 0.49461
valid_loss: 0.00040404
best_epoch: 47
num_days: 233
num_samples: 734349
```

真实执行回测：

```text
IC_mean: 0.01765
ICIR: 0.15497
RankIC_mean: -0.01975
RankICIR: -0.12165
DirectionAcc: 0.50033
num_days: 233
num_samples: 733855
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| DLinear seq10 Top10 Full | 66.70% | 73.80% | 2.35 | -15.34% | 58.37% | 1.0421 |
| DLinear seq10 Buffer-Risk | 53.14% | 58.56% | 1.82 | -14.09% | 59.23% | 0.5090 |
| DLinear seq10 Top10-Drop2 | 48.13% | 52.95% | 1.80 | -17.61% | 57.94% | 0.4026 |
| Random Top10-Drop2 | 51.62% +- 21.62% | 56.99% +- 24.24% | 2.31 +- 0.74 | -9.24% +- 2.01% | 58.20% +- 2.41% | 0.4026 |
| Random Top10 Full | 40.27% +- 11.13% | 44.23% +- 12.37% | 1.96 +- 0.44 | -9.78% +- 2.06% | 56.57% +- 2.47% | 1.9882 |

### DLinear seq20 e60

config: `configs/config_dlinear_seq20_e60.json`

```text
seq_len = 20
moving_avg = 5
epochs = 60
batch_size = 65536
lr = 0.001
weight_decay = 0.0001
dropout = 0.1
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.02205
ICIR: 0.19270
RankIC_mean: 0.00832
RankICIR: 0.05791
DirectionAcc: 0.49759
valid_loss: 0.00040063
best_epoch: 26
num_days: 223
num_samples: 702719
```

真实执行回测：

```text
IC_mean: 0.00547
ICIR: 0.08494
RankIC_mean: -0.00846
RankICIR: -0.07856
DirectionAcc: 0.50223
num_days: 223
num_samples: 702280
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| DLinear seq20 Buffer-Risk | 34.83% | 40.17% | 1.59 | -14.01% | 55.16% | 0.4430 |
| DLinear seq20 Top10-Drop2 | 32.30% | 37.20% | 1.50 | -16.58% | 54.26% | 0.4027 |
| DLinear seq20 Top10 Full | 27.63% | 31.75% | 1.41 | -16.82% | 52.91% | 1.0332 |
| Random Top10-Drop2 | 54.79% +- 14.90% | 63.94% +- 17.85% | 2.65 +- 0.57 | -9.29% +- 1.90% | 58.86% +- 2.35% | 0.4027 |
| Random Top10 Full | 34.11% +- 10.56% | 39.39% +- 12.42% | 1.84 +- 0.45 | -10.48% +- 1.89% | 56.46% +- 2.37% | 1.9887 |

## B. label_oo_1d 严格 T+1 实验结果

### MLP oo

config: `configs/config_mlp_oo.json`

```text
seq_len = N/A
hidden_dims = [512, 256, 128]
epochs = 60
batch_size = 65536
lr = 0.002
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.04442
ICIR: 0.26184
RankIC_mean: 0.03121
RankICIR: 0.15336
DirectionAcc: 0.50919
valid_loss: 0.00049154
best_epoch: 30
num_days: 242
num_samples: 762816
```

真实执行回测：

```text
IC_mean: 0.04019
ICIR: 0.33985
RankIC_mean: 0.03096
RankICIR: 0.24566
DirectionAcc: 0.50497
num_days: 241
num_samples: 758250
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| MLP oo Buffer-Risk | 71.61% | 75.89% | 1.81 | -19.94% | 50.62% | 0.5900 |
| MLP oo Top10 Full | 64.07% | 67.83% | 1.61 | -20.39% | 53.11% | 1.5253 |
| MLP oo Top10-Drop2 | 48.92% | 51.66% | 1.30 | -16.57% | 52.70% | 0.4025 |
| Random Top10-Drop2 | 34.08% +- 12.05% | 35.92% +- 12.76% | 1.22 +- 0.42 | -17.63% +- 2.79% | 54.27% +- 1.95% | 0.4025 |
| Random Top10 Full | 18.09% +- 13.28% | 19.02% +- 13.97% | 0.64 +- 0.52 | -18.69% +- 2.63% | 52.76% +- 1.53% | 1.9899 |

### LSTM seq10 oo e60

config: `configs/config_lstm_seq10_oo_e60.json`

```text
seq_len = 10
hidden_size = 128
num_layers = 2
bidirectional = false
epochs = 60
batch_size = 32768
lr = 0.001
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.03963
ICIR: 0.27497
RankIC_mean: 0.04066
RankICIR: 0.23029
DirectionAcc: 0.52437
valid_loss: 0.00049693
best_epoch: 44
num_days: 233
num_samples: 734349
```

真实执行回测：

```text
IC_mean: 0.03581
ICIR: 0.42440
RankIC_mean: 0.04112
RankICIR: 0.36960
DirectionAcc: 0.51145
num_days: 232
num_samples: 729872
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM seq10 oo Top10 Full | 91.17% | 102.15% | 2.60 | -14.89% | 61.64% | 1.5302 |
| LSTM seq10 oo Buffer-Risk | 54.79% | 60.74% | 1.97 | -14.53% | 54.31% | 0.5983 |
| LSTM seq10 oo Top10-Drop2 | 43.71% | 48.27% | 1.62 | -16.58% | 56.47% | 0.4026 |
| Random Top10-Drop2 | 27.63% +- 14.48% | 30.42% +- 16.12% | 1.23 +- 0.49 | -17.92% +- 2.03% | 54.18% +- 2.47% | 0.4026 |
| Random Top10 Full | 12.10% +- 12.20% | 13.27% +- 13.41% | 0.63 +- 0.52 | -18.26% +- 2.64% | 51.83% +- 2.92% | 1.9890 |

### LSTM seq20 oo e60

config: `configs/config_lstm_seq20_oo_e60.json`

```text
seq_len = 20
hidden_size = 128
num_layers = 2
bidirectional = false
epochs = 60
batch_size = 16384
lr = 0.001
weight_decay = 0.0001
dropout = 0.2
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.03548
ICIR: 0.25599
RankIC_mean: 0.03500
RankICIR: 0.21176
DirectionAcc: 0.51836
valid_loss: 0.00049589
best_epoch: 23
num_days: 223
num_samples: 702719
```

真实执行回测：

```text
IC_mean: 0.02989
ICIR: 0.37405
RankIC_mean: 0.03267
RankICIR: 0.32104
DirectionAcc: 0.50371
num_days: 222
num_samples: 698348
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM seq20 oo Top10 Full | 77.94% | 92.35% | 2.62 | -14.59% | 58.56% | 1.4748 |
| LSTM seq20 oo Top10-Drop2 | 57.04% | 66.92% | 2.22 | -15.11% | 56.31% | 0.4027 |
| LSTM seq20 oo Buffer-Risk | 47.71% | 55.70% | 2.07 | -16.25% | 59.91% | 0.5928 |
| Random Top10-Drop2 | 19.32% +- 11.58% | 22.29% +- 13.43% | 1.01 +- 0.47 | -19.04% +- 2.96% | 54.08% +- 2.10% | 0.4027 |
| Random Top10 Full | 5.59% +- 12.96% | 6.48% +- 14.89% | 0.42 +- 0.59 | -19.17% +- 3.27% | 52.84% +- 2.54% | 1.9894 |

### DLinear seq10 oo e60

config: `configs/config_dlinear_seq10_oo_e60.json`

```text
seq_len = 10
moving_avg = 3
epochs = 60
batch_size = 65536
lr = 0.001
weight_decay = 0.0001
dropout = 0.1
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.00341
ICIR: 0.03307
RankIC_mean: 0.00419
RankICIR: 0.02953
DirectionAcc: 0.48527
valid_loss: 0.00050566
best_epoch: 49
num_days: 233
num_samples: 734349
```

真实执行回测：

```text
IC_mean: 0.00868
ICIR: 0.12944
RankIC_mean: 0.01622
RankICIR: 0.16189
DirectionAcc: 0.49135
num_days: 232
num_samples: 729872
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| DLinear seq10 oo Top10-Drop2 | 35.52% | 39.11% | 1.39 | -16.96% | 54.31% | 0.4026 |
| DLinear seq10 oo Buffer-Risk | 29.53% | 32.45% | 1.16 | -19.48% | 55.60% | 0.5664 |
| DLinear seq10 oo Top10 Full | 10.79% | 11.77% | 0.52 | -22.22% | 54.31% | 1.1440 |
| Random Top10-Drop2 | 27.63% +- 14.48% | 30.42% +- 16.12% | 1.23 +- 0.49 | -17.92% +- 2.03% | 54.18% +- 2.47% | 0.4026 |
| Random Top10 Full | 12.10% +- 12.20% | 13.27% +- 13.41% | 0.63 +- 0.52 | -18.26% +- 2.64% | 51.83% +- 2.92% | 1.9890 |

### DLinear seq20 oo e60

config: `configs/config_dlinear_seq20_oo_e60.json`

```text
seq_len = 20
moving_avg = 5
epochs = 60
batch_size = 65536
lr = 0.001
weight_decay = 0.0001
dropout = 0.1
early_stop_patience = 8
```

验证集结果：

```text
IC_mean: 0.02256
ICIR: 0.19490
RankIC_mean: 0.02775
RankICIR: 0.20509
DirectionAcc: 0.48923
valid_loss: 0.00050241
best_epoch: 26
num_days: 223
num_samples: 702719
```

真实执行回测：

```text
IC_mean: 0.01186
ICIR: 0.12960
RankIC_mean: 0.02064
RankICIR: 0.20806
DirectionAcc: 0.49131
num_days: 222
num_samples: 698348
```

| 策略 | 总收益 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| DLinear seq20 oo Top10 Full | 18.43% | 21.17% | 0.96 | -20.74% | 49.55% | 1.0225 |
| DLinear seq20 oo Buffer-Risk | 8.78% | 10.03% | 0.62 | -21.66% | 51.80% | 0.4604 |
| DLinear seq20 oo Top10-Drop2 | 6.84% | 7.80% | 0.48 | -21.51% | 53.15% | 0.4027 |
| Random Top10-Drop2 | 19.32% +- 11.58% | 22.29% +- 13.43% | 1.01 +- 0.47 | -19.04% +- 2.96% | 54.08% +- 2.10% | 0.4027 |
| Random Top10 Full | 5.59% +- 12.96% | 6.48% +- 14.89% | 0.42 +- 0.59 | -19.17% +- 3.27% | 52.84% +- 2.54% | 1.9894 |

## C. 两套标签对比

### 标签定义关系

```text
label_oc_1d = close[t+1] / open[t+1] - 1
label_oo_1d = open[t+2]  / open[t+1] - 1
```

二者的共同点：

```text
1. 都在第 t 日收盘后生成信号。
2. 都假设第 t+1 日开盘买入。
3. 都避免了 close-to-close 标签中“收盘后才有信号，却按当日收盘成交”的执行问题。
```

二者的区别：

```text
1. label_oc_1d 是第 t+1 日开盘到第 t+1 日收盘的日内收益。
2. label_oo_1d 是第 t+1 日开盘到第 t+2 日开盘的 open-to-open 收益。
3. label_oo_1d 更符合 A 股 T+1：买入当天不能卖出，至少要持有到下一交易日。
4. label_oc_1d 更像持仓当日 mark-to-market，label_oo_1d 更像可实际落地的卖出收益。
```

### OC vs OO 标签相关性分析

相关性分析结果目录为：

```text
outputs_benchmark/label_correlation/
```

其中整体统计、每日横截面 summary、年度统计和月度统计分别见：

```text
oc_oo_overall_correlation.csv
oc_oo_daily_correlation_summary.csv
oc_oo_yearly_correlation.csv
oc_oo_monthly_correlation.csv
```

整体样本覆盖 `7,462,146` 条记录、`2,482` 个交易日和 `3,163` 只股票。两套标签在整体样本和每日横截面上都保持较高相关性：

| 统计口径 | Pearson | Spearman | 样本补充 |
|---|---:|---:|---|
| 全样本 | 0.84468 | 0.91707 | 7,462,146 条记录 |
| 每日横截面均值 | 0.83824 | 0.90766 | 2,482 个交易日 |
| 每日横截面标准差 | 0.09349 | 0.03112 | 相关性日度波动 |

标签分布仍存在明显差异：

| 指标 | `label_oc_1d` | `label_oo_1d` |
|---|---:|---:|
| 均值 | 0.001205 | 0.000371 |
| 标准差 | 0.026323 | 0.032381 |
| P01 | -0.067288 | -0.079545 |
| P50 | 0.000000 | 0.000000 |
| P99 | 0.088095 | 0.099950 |

相关性结论：

```text
1. OC 与 OO 标签高度相关，说明二者刻画的是同一类短周期收益信号，排序关系尤其接近。
2. Spearman 高于 Pearson，说明横截面排序关系比收益幅度关系更稳定。
3. OO 的标准差和尾部分位绝对值更大，说明隔夜持有引入了更宽的收益分布和额外风险。
4. 因为交易时点不同，OC 不能直接替代严格 T+1 的 OO；相关性高不等于回测口径等价。
5. 相关性分析支持保留 OC 作为对照标签，同时继续以 OO 作为真实交易主口径。
```

### 真实执行预测指标对比

| 模型 | OC IC_mean | OC ICIR | OC RankIC | OO IC_mean | OO ICIR | OO RankIC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.02265 | 0.18847 | -0.01660 | 0.04019 | 0.33985 | 0.03096 |
| LSTM seq10 | 0.02340 | 0.26272 | -0.01122 | 0.03581 | 0.42440 | 0.04112 |
| LSTM seq20 | 0.02179 | 0.24889 | -0.01131 | 0.02989 | 0.37405 | 0.03267 |
| DLinear seq10 | 0.01765 | 0.15497 | -0.01975 | 0.00868 | 0.12944 | 0.01622 |
| DLinear seq20 | 0.00547 | 0.08494 | -0.00846 | 0.01186 | 0.12960 | 0.02064 |

### 策略收益对比

| 模型 | OC 最佳策略 | OC 最佳总收益 | OO 最佳策略 | OO 最佳总收益 |
|---|---|---:|---|---:|
| MLP | Top10 Full | 130.33% | Buffer-Risk | 71.61% |
| LSTM seq10 | Top10 Full | 183.77% | Top10 Full | 91.17% |
| LSTM seq20 | Top10 Full | 129.27% | Top10 Full | 77.94% |
| DLinear seq10 | Top10 Full | 66.70% | Top10-Drop2 | 35.52% |
| DLinear seq20 | Buffer-Risk | 34.83% | Top10 Full | 18.43% |

### 主要观察

```text
1. OC 与 OO 标签全样本 Pearson 为 0.84468、Spearman 为 0.91707，说明标签高度相关，但交易口径并不等价。
2. OO 标签收益分布更宽，标准差与上下尾部分位都大于 OC，符合隔夜持有带来额外波动的直觉。
3. OO 口径下 MLP 和 LSTM 的 IC、ICIR、RankIC、RankICIR 全面改善。
4. OO 口径下策略总收益普遍低于 OC 口径，这是合理的，因为持仓周期跨到下一交易日开盘，暴露了隔夜风险。
5. OC 口径的收益更高，但包含“当天收盘估值”的 mark-to-market 成分；OO 口径更接近真实可成交收益。
6. LSTM seq10 在两套标签下都很强，是目前最稳定的深度学习主模型。
7. LSTM seq20 在 OO 口径下与 seq10 的差距缩小，说明更长窗口对 open-to-open 收益有一定帮助。
8. DLinear seq10 的验证集结果更正后，IC_mean 从旧记录 0.02205 降至 0.01889，RankIC_mean 由正转负；但真实执行回测和策略表现未改变。
9. DLinear seq20 的真实执行回测仍弱于 MLP/LSTM，且 seq20 的 OC 策略收益低于 seq10，说明拉长 DLinear 窗口没有带来稳定提升。
10. DLinear 在 OC 和 OO 两套口径下都弱于 MLP/LSTM，适合作为线性序列 baseline，不适合作为当前主策略。
11. OO 实验中随机策略收益显著下降，模型策略仍能超过随机策略，说明严格 T+1 下模型信号仍有交易价值。
```

## D. 市场基准与交易成本敏感性

### 市场指数基准

指数基准由 `src/benchmark_index_local.py` 生成，结果文件为：

```text
outputs_benchmark/index/index_benchmark_metrics.csv
```

当前基准默认按指数 close-to-close 日收益计算 2025 年净值。

| 指数 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 | 天数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 上证指数 | 21.65% | 22.53% | 13.25% | 1.60 | -9.71% | 57.20% | 243 |
| 沪深300 | 21.19% | 22.06% | 14.96% | 1.41 | -10.49% | 55.56% | 243 |
| 创业板指 | 55.46% | 58.02% | 29.10% | 1.72 | -20.79% | 53.91% | 243 |

和严格 T+1 候选策略相比：

```text
1. LSTM seq10 oo + Top10 Full 总收益 91.17%，高于三类指数基准。
2. MLP oo + Buffer-Risk 总收益 71.61%，也高于三类指数基准。
3. LSTM seq10 oo + Top10-Drop2 总收益 43.71%，高于上证指数和沪深300，但低于创业板指。
4. 指数基准说明模型收益不只是和随机策略比较；在 2025 年上涨市场中，严格 T+1 主候选仍有超越主要宽基指数的表现。
```

### 手续费 / 滑点敏感性

交易成本敏感性由 `src/fee_sensitivity.py` 生成，结果文件为：

```text
outputs_benchmark/fee_sensitivity/fee_sensitivity_metrics.csv
```

当前实验将 `fee_rate` 从 `0.0003` 提高到 `0.003`，用于近似观察手续费、滑点和交易冲击增大后的策略退化。

| 策略 | fee_rate | 总收益 | 年化收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM seq10 oo Top10 Full | 0.0003 | 91.17% | 102.15% | 2.60 | -14.89% | 1.5302 |
| LSTM seq10 oo Top10 Full | 0.0010 | 49.30% | 54.54% | 1.65 | -16.83% | 1.5302 |
| LSTM seq10 oo Top10 Full | 0.0020 | 4.82% | 5.25% | 0.30 | -20.99% | 1.5302 |
| LSTM seq10 oo Top10 Full | 0.0030 | -26.44% | -28.36% | -1.05 | -31.74% | 1.5302 |
| LSTM seq10 oo Top10-Drop2 | 0.0003 | 43.71% | 48.27% | 1.62 | -16.58% | 0.4026 |
| LSTM seq10 oo Top10-Drop2 | 0.0010 | 34.72% | 38.22% | 1.35 | -17.37% | 0.4026 |
| LSTM seq10 oo Top10-Drop2 | 0.0030 | 12.00% | 13.10% | 0.58 | -19.60% | 0.4026 |
| MLP oo Buffer-Risk | 0.0003 | 71.61% | 75.89% | 1.81 | -19.94% | 0.5900 |
| MLP oo Buffer-Risk | 0.0010 | 55.48% | 58.64% | 1.47 | -20.46% | 0.5900 |
| MLP oo Buffer-Risk | 0.0030 | 17.25% | 18.10% | 0.51 | -21.95% | 0.5900 |

成本敏感性结论：

```text
1. Top10 Full 的收益最高，但平均换手 1.5302，对交易成本最敏感。
2. Top10 Full 在 fee_rate=0.002 时总收益只剩 4.82%，在 fee_rate=0.003 时转负。
3. LSTM seq10 oo + Top10-Drop2 的低换手优势明显，高成本下仍保留正收益。
4. MLP oo + Buffer-Risk 在 fee_rate=0.001 时总收益仍为 55.48%，接近创业板指基准且明显高于上证指数、沪深300。
5. 最终模拟交易不宜只按低费率 Top10 Full 的总收益选择策略，需要同时看换手率与成本敏感性。
```

## E. 最终模型结论

| 目标 | 推荐方案 | 理由 |
|---|---|---|
| 研究展示收益最高 | LSTM seq10 + OC + Top10 Full | OC 总收益最高 183.77%，但平均换手 1.6489 |
| 收益-换手均衡 | LSTM seq10 + OC + Top10-Drop2 | 总收益 172.62%，平均换手 0.4026 |
| 严格 T+1 低成本收益最高 | LSTM seq10 oo + Top10 Full | OO 总收益最高 91.17%，但成本敏感 |
| 严格 T+1 收益-成本均衡 | LSTM seq10 oo + Top10-Drop2 | 换手 0.4026，高费率下退化更慢 |
| 严格 T+1 稳健策略 | MLP oo + Buffer-Risk | 收益高于指数基准，逻辑更稳健 |
| 线性 baseline | DLinear seq10 / seq20 | 明显弱于 MLP/LSTM，适合作为对照 |

## F. 当前推荐

OC 收益最高：

```text
LSTM seq10 e60 + label_oc_1d + Top10 Full
```

原因：

```text
总收益 183.77%
年化收益 208.96%
Sharpe 3.97
最大回撤 -13.89%
平均换手 1.6489
说明：收益最高，但换手率较高，对手续费、滑点和成交冲击更敏感。
```

收益-换手均衡：

```text
LSTM seq10 e60 + label_oc_1d + Top10-Drop2
```

原因：

```text
总收益 172.62%
年化收益 195.85%
Sharpe 3.85
最大回撤 -11.36%
平均换手 0.4026
```

严格 T+1 收益最高：

```text
LSTM seq10 oo e60 + label_oo_1d + Top10 Full
```

原因：

```text
严格 T+1 open-to-open 口径
总收益 91.17%
年化收益 102.15%
Sharpe 2.60
最大回撤 -14.89%
胜率 61.64%
平均换手 1.5302
说明：严格 T+1 口径下收益最高，但换手率较高，最终模拟交易需要结合手续费/滑点敏感性结果审慎选择。
```

严格 T+1 收益-成本均衡：

```text
LSTM seq10 oo e60 + label_oo_1d + Top10-Drop2
```

原因：

```text
严格 T+1 open-to-open 口径
fee_rate=0.0003 时总收益 43.71%
fee_rate=0.0010 时总收益 34.72%
fee_rate=0.0030 时总收益 12.00%
平均换手 0.4026
收益低于 Top10 Full，但对交易成本更不敏感。
```

严格 T+1 稳健备选：

```text
MLP oo + label_oo_1d + Buffer-Risk
```

原因：

```text
严格 T+1 open-to-open 口径
总收益 71.61%
年化收益 75.89%
最大回撤 -19.94%
平均换手 0.5900
策略逻辑更稳健、更容易解释
```

## G. 最终模拟交易操作方法

比赛阶段采用 `label_oo_1d` 的严格 T+1 口径，默认使用低换手方案：

```text
主模型：LSTM seq10 oo e60
目标持仓：Top10
每日换仓：Drop2
交易原则：第 t 日盘后生成信号，第 t+1 日交易时间人工下单
```

### 图形界面操作流程

比赛期间优先使用 Streamlit 图形化助手，减少手动拼命令和漏更新持仓的概率。

1. 先安装项目依赖。图形界面额外依赖 `streamlit`：

```powershell
pip install -r requirements.txt
```

2. 启动交易助手。若脚本位于仓库 `src/` 目录，运行：

```powershell
streamlit run src/trading_dashboard.py
```

若当前文件位于项目根目录，则运行：

```powershell
streamlit run trading_dashboard.py
```

3. 在侧边栏核对项目根目录、`predict_daily.py`、`make_trade_plan.py`、当前持仓文件、checkpoint、训练期 scaler、`TopK=10`、`DropK=2`、`signal_date` 和计划交易日期。
4. 先看页面顶部“比赛执行预检”。脚本、模型输入、候选日期关系或股票池提示存在异常时，先修正再生成交易计划。
5. 在“上传/选择数据”页上传最新 `csv`、`parquet` 或 `zip` 行情数据，或者选择已经同步好的行情数据目录。
6. 在“当前持仓”页维护同花顺模拟盘真实持仓。最少保留 `ts_code`，建议同时记录 `buy_date`、`shares`、`weight`；成交后也在这里更新。
7. 在“生成预测”页运行每日预测，得到 `outputs_daily/latest_candidates.csv`，并检查 Top 50 候选分数和信号日期。
8. 在“生成交易计划”页生成比赛交易计划，核对 Sell、Buy、Hold 数量以及完整计划表。
9. 在“查看与下载”页下载 `buy_list`、`sell_list`、`hold_list` 和 `trade_plan`，按清单到同花顺模拟盘人工执行。
10. 成交完成后返回“当前持仓”页更新实际持仓，保存当日候选清单、交易计划、持仓截图、成交截图和收益截图。

### 命令行备用流程

当界面环境不可用时，可以直接运行脚本完成同一条链路。

1. 每个交易日盘后同步最新 A 股日频数据，确认输入数据只到本次 `signal_date`，不使用未来交易日字段。
2. 使用训练好的 LSTM checkpoint 和训练期 scaler 运行每日预测脚本，生成最新候选分数：

```powershell
python src/predict_daily.py `
  --data_dir <latest_stock_data_dir> `
  --checkpoint <lstm_seq10_oo_checkpoint> `
  --scaler <train_scaler_json> `
  --signal_date 2026-06-01 `
  --trade_date 2026-06-02 `
  --seq_len 10 `
  --top_k_hold 10 `
  --drop_k 2
```

3. 根据同花顺模拟盘真实成交后的持仓维护 `data/current_positions.csv`。最少保留 `ts_code`，建议同时记录 `buy_date`、`shares`、`weight`，使调仓计划更容易核对。
4. 使用最新候选分数和当前持仓生成正式比赛交易计划：

```powershell
python src/make_trade_plan.py `
  --pred outputs_daily/latest_candidates.csv `
  --positions data/current_positions.csv `
  --trade_date 2026-06-02 `
  --topk 10 `
  --dropk 2
```

5. 在 `outputs_trade_plan/` 中依次核对 `latest_sell_list.csv`、`latest_buy_list.csv`、`latest_hold_list.csv` 和 `latest_trade_plan.csv`。
6. 比赛首日若还没有持仓，按买入清单等权建仓 10 只股票，尽可能使用全部现金保持满仓。
7. 后续交易日先执行卖出清单，再执行买入清单，目标是卖出当前持仓中模型分数最低的 2 只，并补入未持有候选中的高分股票，使收盘后仍维持 Top10 附近满仓状态。
8. 每次下单后根据实际成交情况更新 `data/current_positions.csv`。如果委托未成交，需要在交易时间内撤单、改价或重新下单，最终记录真实成交而不是只记录计划。

### 交易约束与留痕

```text
1. 严格遵守 A 股 T+1：当日新买入股票不在当日再次卖出。
2. 每日股票池必须排除北交所与 ST 股票；`predict_daily.py` 会过滤 `.BJ`，ST 过滤依赖输入数据中存在可识别的 `name` 字段，比赛前需额外核对候选清单。
3. `--trade_date` 建议显式传入；遇到周末或节假日时不要依赖脚本默认的自然日 +1。
4. 比赛要求每日满仓，计划文件只给出目标权重，最终要结合现金、股数和实际成交手工完成建仓或补仓。
5. 若某只持仓在候选池里缺失，先核对是否停牌、数据缺失或被股票池过滤，再决定是否按默认计划优先卖出。
6. 图形界面预检是操作辅助，不替代人工核对；下单前仍要确认交易日期、股票池、现金占用和真实成交状态。
7. 每日保存候选清单、交易计划、实际调仓记录、持仓截图和收益截图，赛后用于模拟交易分析与报告附件。
```

按上述流程，项目从盘后数据、模型预测、候选排序、交易计划到模拟盘人工执行已经闭环；比赛结果分析阶段只需补充真实收益、调仓截图和持仓截图。

## H. 后续操作

```text
已完成：
   市场指数基准。
   手续费 / 滑点近似敏感性实验。
   OC vs OO 标签相关性分析。
   每日实盘候选脚本。
   比赛交易计划脚本。
   图形化交易助手。

第一优先级：最终 LaTeX 报告
   将 README 中的结果整理为论文式结构，包括实验设置、模型结构、结果分析、消融对比和反思。

最终报告口径：
   优先使用 label_oo_1d 作为主交易口径。
   label_oc_1d 可以保留为 mark-to-market 对照实验。

主模型：
   继续保留 LSTM seq10 作为主模型。
   它在 OC 和 OO 两套标签下都表现稳定，适合作为最终策略核心。

比赛期间：
   记录同花顺模拟盘截图。
   包括收益趋势、调仓记录、持仓记录和每日模型推荐清单。
```

## I. GitHub 上传说明

不上传大型数据文件和模型权重。

建议上传：

```text
源代码
配置文件
README.md
requirements.txt
src/trading_dashboard.py
src/predict_daily.py
src/make_trade_plan.py
推理用 scaler.json 或与权重同包提供的 scaler.json
关键结果图
小型 JSON 指标
strategy_metrics_summary*.csv
monthly_ic*.csv
```

不建议上传：

```text
原始数据
处理后的 parquet 数据
模型权重
大型预测 CSV
大型持仓记录
大型交易记录
```

若希望其他人下载后直接使用图形界面，发布前至少确认：

```text
1. GitHub 中存在 README.md。
2. GitHub 中存在 requirements.txt，且包含 streamlit。
3. GitHub 中存在 src/trading_dashboard.py、src/predict_daily.py、src/make_trade_plan.py。
4. 权重下载包中同时提供 best_model.pt 与配套 scaler.json。
5. README 说明行情数据需要由使用者本地准备，不能只下载权重就直接预测。
```
