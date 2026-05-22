# 图形化预测交易操作手册

本文只说明如何在本地使用仓库、模型权重和图形化程序完成每日预测、生成交易计划、执行模拟交易并维护持仓。

## 1. 第一次使用前准备

### 1.1 手动安装基础软件

请先在本机确认已经安装：

1. `Git`
2. `Git LFS`
3. `Python 3.10+` 或 Conda 环境
4. 能访问 GitHub 的网络环境

模型权重通过 Git LFS 管理。缺少 Git LFS 时，权重文件可能只会拉到几行指针文本，程序无法正常加载。

### 1.2 手动克隆仓库

在你准备存放项目的目录打开 PowerShell，执行：

```powershell
git lfs install
git clone https://github.com/cbyybc/deeplearning.git
cd deeplearning
git lfs pull
```

### 1.3 手动确认关键文件存在

克隆完成后，在仓库根目录确认下列文件存在：

```text
requirements.txt
src/trading_dashboard.py
src/predict_daily.py
src/make_trade_plan.py
outputs_lstm_seq10_oo_e60/models/best_lstm.pt
outputs_lstm_seq10_oo_e60/preprocess_state_lstm.json
```

模型权重默认路径：

```text
outputs_lstm_seq10_oo_e60/models/best_lstm.pt
```

训练期预处理参数默认路径：

```text
outputs_lstm_seq10_oo_e60/preprocess_state_lstm.json
```

## 2. 安装 Python 依赖

### 2.1 手动进入项目环境

如果使用 Conda，可以先创建或激活环境，例如：

```powershell
conda create -n dltrade python=3.11
conda activate dltrade
```

如果你已经有可用 Python 环境，也可以直接使用现有环境。

### 2.2 手动安装依赖

在仓库根目录执行：

```powershell
pip install -r requirements.txt
```

如果之后运行 dashboard 提示没有 `streamlit`，重新执行上面的安装命令。

## 3. 启动图形化程序

### 3.1 手动启动 dashboard

在仓库根目录执行：

```powershell
streamlit run src/trading_dashboard.py
```

Streamlit 会输出一个本地地址，通常类似：

```text
http://localhost:8501
```

在浏览器打开该地址。

### 3.2 第一次打开后手动核对侧边栏

请先在左侧侧边栏逐项核对：

| 项目 | 推荐值 |
|---|---|
| 项目根目录 | 当前 clone 下来的仓库根目录 |
| 每日预测脚本 | `src/predict_daily.py` |
| 交易计划脚本 | `src/make_trade_plan.py` |
| 当前持仓文件 | `data/current_positions.csv` |
| 预测输出目录 | `outputs_daily` |
| 交易计划输出目录 | `outputs_trade_plan` |
| 目标持仓 TopK | `10` |
| 每日换仓 DropK | `2` |
| 模型权重 checkpoint | `outputs_lstm_seq10_oo_e60/models/best_lstm.pt` |
| 训练集预处理参数 | `outputs_lstm_seq10_oo_e60/preprocess_state_lstm.json` |
| `seq_len` | `10` |
| `device` | 有可用 CUDA 可选 `cuda`，否则选 `cpu` |

## 4. 准备本地累计历史行情目录

预测程序不能只读取当天一张 CSV。它需要历史日行情计算滚动特征，并构造 LSTM 最近 `seq_len=10` 的序列输入。

### 4.1 累计目录位置

dashboard 默认使用：

```text
data/local_market_data/
```

该目录只保留在本地，已经被 `.gitignore` 忽略，不会随代码上传到 GitHub。

### 4.2 行情文件最低字段要求

每个日行情 CSV 或 Parquet 至少需要包含：

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

日行情 CSV 表头示例：

```csv
ts_code,trade_date,open,high,low,close,pre_close,vol,amount
000001.SZ,20260521,10.78,10.80,10.69,10.70,10.77,1110211.08,1194144.112
```

不要把股票状态表、ST 列表或只含 `type/type_name` 的文件当成日行情上传。

### 4.3 第一次手动建立历史底座

第一次使用时，需要让累计目录中有足够历史交易日。建议至少准备最近 30 个以上交易日，实际使用中可以准备更长历史。

你可以二选一：

1. 在 dashboard 的“上传/选择数据”页上传一份含历史日行情的 `zip`。
2. 手动把历史日行情 CSV/Parquet 文件复制到：

   ```text
   data/local_market_data/
   ```

第一次准备完成后，目录形态可以类似：

```text
data/local_market_data/
  20260313.csv
  20260316.csv
  20260317.csv
  ...
  20260520.csv
  20260521.csv
```

### 4.4 之后每天只追加最新一天

累计目录建立后，日常不需要重复上传旧历史数据。

每天盘后拿到新日行情后，只需要：

1. 在 dashboard 的“上传/选择数据”页上传当天最新 CSV/Parquet，例如 `20260522.csv`。
2. 或手动把当天文件复制到：

   ```text
   data/local_market_data/
   ```

dashboard 上传时会把日行情文件同步进累计目录。只要累计目录里原有历史数据还在，之后每天只追加新交易日即可。

## 5. 第一次创建当前持仓文件

交易计划脚本需要知道你当前真实持有什么股票。

### 5.1 空仓时手动创建空持仓

第一次还没有持仓时，在 dashboard 打开：

```text
2 当前持仓
```

点击：

```text
创建空持仓文件
```

空持仓文件应保留表头：

```csv
ts_code,buy_date,shares,weight
```

### 5.2 已有持仓时手动填写

如果你已经在模拟盘持仓，请在“当前持仓”页填写真实持仓。

示例：

```csv
ts_code,buy_date,shares,weight
002878.SZ,2026-05-22,1000,0.10
600423.SH,2026-05-22,800,0.10
```

字段含义：

| 字段 | 手动填写要求 |
|---|---|
| `ts_code` | 必填，股票代码必须正确 |
| `buy_date` | 建议填写真实买入日期 |
| `shares` | 建议填写真实持股数 |
| `weight` | 建议填写当前或目标仓位比例 |

交易计划主要依赖 `ts_code` 判断当前持仓。`buy_date`、`shares`、`weight` 用于你自己复核和后续记录。

## 6. 每日预测前的手动检查

每天生成预测前，请按顺序完成下面检查。

### 6.1 手动追加当天最新行情

确认当天最新完整日行情已经加入累计目录。

例如你准备预测 `2026-05-22` 信号时，应存在：

```text
data/local_market_data/20260522.csv
```

并确认文件内 `trade_date` 确实是：

```text
20260522
```

文件名叫 5 月 22 日并不够，文件内容中的 `trade_date` 也要正确。

### 6.2 手动确认侧边栏行情目录

侧边栏“行情数据目录”推荐保持：

```text
<仓库根目录>/data/local_market_data
```

不要在日常运行时把它改成只含当天一张 CSV 的临时目录。

### 6.3 手动决定 `signal_date`

日常预测时，`signal_date` 通常可以留空。

留空时预测脚本会自动选择行情目录中最大的 `trade_date` 作为信号日。

只有在以下情况才建议手动填写：

1. 你要复现历史某一天预测。
2. 你怀疑目录混入错误日期。
3. 你明确要锁定某个信号日做检查。

### 6.4 手动设置 `trade_date`

`trade_date` 是你真实执行模拟交易的日期，建议每次手动确认。

本项目执行口径是：

```text
signal_date = t 日盘后信号
trade_date  = 下一交易日执行交易
```

请特别注意：

| 情况 | 正确操作 |
|---|---|
| 周一到周四盘后出信号 | 一般设置为下一自然日交易日 |
| 周五盘后出信号 | 设置为下周一，不能写周六 |
| 节假日前盘后出信号 | 设置为节后首个交易日 |
| 当天文件还不是完整日线 | 不要提前把当天设为信号日 |

生成交易计划前，务必确认：

```text
trade_date > signal_date
```

## 7. 在 dashboard 生成每日预测

### 7.1 手动进入预测页

打开 dashboard：

```text
3 生成预测
```

### 7.2 手动复核命令内容

页面会展示将要调用的预测命令。请重点看：

1. `--data_dir` 是否指向累计行情目录。
2. `--checkpoint` 是否指向默认 LSTM 权重。
3. `--scaler` 是否指向默认预处理文件。
4. `--seq_len` 是否为 `10`。
5. `--trade_date` 是否为下一交易日。

### 7.3 手动点击运行

点击：

```text
运行每日预测
```

成功后会生成：

```text
outputs_daily/latest_candidates.csv
outputs_daily/daily_candidates_<signal_date>.csv
```

页面会显示候选股票数、信号日期和 Top 50 候选分数。

### 7.4 手动检查预测结果

请人工确认：

1. 页面显示的信号日是不是你想要的最新行情日。
2. 候选股票数不是 0。
3. Top 50 候选不是明显异常数据。
4. 没有出现 `.BJ` 股票。
5. 如果候选文件缺少 `name` 列，请额外核对是否还残留不合规 ST 股票。

## 8. 生成交易计划

### 8.1 手动进入交易计划页

打开：

```text
4 生成交易计划
```

### 8.2 手动确认输入

生成计划前确认：

1. 当前持仓文件已经是最新真实持仓。
2. `latest_candidates.csv` 是刚刚生成的新预测结果。
3. `trade_date` 是实际准备下单的交易日。
4. `TopK=10`、`DropK=2` 没被误改。

### 8.3 手动运行计划生成

点击生成计划后，输出目录中会出现：

```text
outputs_trade_plan/latest_buy_list.csv
outputs_trade_plan/latest_sell_list.csv
outputs_trade_plan/latest_hold_list.csv
outputs_trade_plan/latest_trade_plan.csv
```

### 8.4 手动查看计划

进入：

```text
5 查看与下载
```

逐项检查：

1. `sell` 列表：计划卖出的持仓。
2. `buy` 列表：计划新买入的股票。
3. `hold` 列表：计划继续持有的股票。
4. `trade_plan`：完整动作清单。

## 9. 在模拟盘真实执行交易

交易计划只是辅助清单，不会自动替你下单。

### 9.1 到 `trade_date` 当天手动下单

在计划中的 `trade_date` 当天，打开比赛模拟交易平台，按交易计划执行。

### 9.2 手动处理未成交情况

真实操作时要以实际成交为准。

常见情况：

1. 计划买入但涨停或流动性不足，未成交。
2. 计划卖出但跌停或挂单未成交。
3. 实际成交股数与计划目标仓位有偏差。
4. 账户现金约束导致不能完全等权买入。

不要把“计划动作”直接当成“已经成交”。

## 10. 交易后更新当前持仓

这是每天最重要的手动闭环。

### 10.1 更新原则

`data/current_positions.csv` 必须表示模拟盘中当前真实持仓，而不是脚本想象中的持仓。

交易完成后：

1. 实际卖出成功的股票，从持仓中删除。
2. 实际卖出失败的股票，继续保留。
3. 实际买入成功的股票，加入持仓。
4. 实际买入失败的股票，不要加入。
5. 更新真实 `shares`。
6. 更新真实或近似 `weight`。
7. 点击保存。

### 10.2 在 dashboard 中手动更新

回到：

```text
2 当前持仓
```

在表格中按真实成交结果修改，然后点击：

```text
保存当前持仓
```

### 10.3 第二天预测前再次核对

第二天开始前，先打开当前持仓页确认持仓仍和模拟盘一致，再生成新交易计划。

## 11. 每日完整操作清单

每天可以按下面顺序执行。

### 11.1 盘后准备

1. 获取当天最新完整日行情 CSV/Parquet。
2. 将当天行情上传到 dashboard，或复制进 `data/local_market_data/`。
3. 确认累计目录中历史数据仍在。
4. 打开 dashboard。
5. 核对当前持仓文件。
6. `signal_date` 留空或手动锁定信号日。
7. 手动设置下一交易日 `trade_date`。
8. 运行每日预测。
9. 检查信号日和候选股票。
10. 生成交易计划。
11. 下载或查看买入、卖出、持有清单。

### 11.2 交易日执行

1. 到 `trade_date` 当天打开模拟盘。
2. 根据计划人工下单。
3. 记录实际成交与未成交情况。
4. 交易后在 dashboard 更新当前持仓。
5. 保存 `data/current_positions.csv`。

## 12. 常见错误处理

### 12.1 缺少 `close`、`open` 等列

说明上传的不是日行情文件，或字段不完整。

手动检查：

1. CSV 是否包含 `open/high/low/close/pre_close/vol/amount`。
2. 是否误传了 ST 列表、股票池状态表或别的业务表。

### 12.2 `No valid sequence generated`

说明没有股票成功构造最终输入序列。

手动检查：

1. 当前行情目录是否只放了当天一张 CSV。
2. 累计目录是否有至少最近 30 个以上交易日。
3. 最新信号日数据是否真的在目录中。
4. 最新 CSV 内 `trade_date` 是否正确。

### 12.3 信号日还是旧日期

手动检查：

1. 最新 CSV 是否已经同步到 `data/local_market_data/`。
2. 最新 CSV 内 `trade_date` 是否是目标日期。
3. 预测是否已经重新运行成功。
4. 页面显示的候选文件是否还是旧的 `outputs_daily/latest_candidates.csv`。

### 12.4 模型权重加载报错

手动检查：

1. 是否执行过 `git lfs pull`。
2. checkpoint 是否是仓库默认权重。
3. 本地代码是否和 checkpoint 属于同一版仓库。

### 12.5 当前持仓编辑页报类型错误

先确认你已经使用最新 dashboard 代码。当前版本会把空 `buy_date` 按文本列处理，避免空日期被 pandas 读成浮点列。

## 13. 推荐的手动记录

比赛期间建议每天保留：

1. 当天上传的最新行情日期。
2. 当天信号日。
3. 实际交易日。
4. `latest_candidates.csv`。
5. `latest_trade_plan.csv`。
6. 实际成交记录。
7. 更新后的 `current_positions.csv`。
8. 模拟盘收益、持仓和交易截图。

这些记录可用于最终报告、答辩复盘和比赛结果说明。
