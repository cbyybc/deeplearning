import os
import glob
import gc
import warnings
from bisect import bisect_right

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================

BASIC_PATH = "../Datasets/basic.csv"
DAILY_DIR = "../Datasets/daily"
STOCK_ST_DIR = "../Datasets/stock_st"

SAVE_ROOT = "Datasets"
PROCESSED_DIR = os.path.join(SAVE_ROOT, "processed")
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

BASIC_FILTERED_PATH = os.path.join(SAVE_ROOT, "basic_filtered.csv")
FEATURE_PARQUET_PATH = os.path.join(PROCESSED_DIR, "all_stock_features.parquet")
FEATURE_PICKLE_PATH = os.path.join(PROCESSED_DIR, "all_stock_features.pkl")
FEATURE_CSV_PATH = os.path.join(PROCESSED_DIR, "all_stock_features.csv")


# ============================================================
# 1. 股票池过滤
# ============================================================

def filter_stock(basic_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤股票池：
    1. 剔除北交所
    2. 剔除当前名称含 ST / 退市的股票
    3. 剔除 2019-01-01 之后上市的股票
    """
    df = basic_df.copy()

    # 剔除北交所
    if "market" in df.columns:
        df = df[~df["market"].astype(str).str.contains("北交所", na=False)]

    # 剔除当前名称含 ST / 退市的股票。历史每日 ST 状态会在读取 daily 时继续按 stock_st 过滤。
    if "name" in df.columns:
        df = df[~df["name"].astype(str).str.contains(r"ST|\*ST|退市", case=False, na=False, regex=True)]

    # 过滤上市时间
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(
            df["list_date"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )
        df = df[df["list_date"] < "2019-01-01"]

    df = df.dropna(subset=["ts_code"])
    df = df.drop_duplicates(subset=["ts_code"])
    df = df.reset_index(drop=True)

    return df


def load_stock_st_by_date(stock_st_dir: str) -> tuple[dict[int, set[str]], list[int]]:
    """
    读取每日 stock_st 列表，用于逐交易日过滤 ST 股票。

    stock_st 数据自 2016 年 8 月起提供。对于任一 daily 交易日，只使用
    trade_date <= 当前交易日的最新 ST 列表，避免用到未来状态。
    """
    if not os.path.exists(stock_st_dir):
        raise FileNotFoundError(
            f"找不到 stock_st 目录: {stock_st_dir}\n"
            "请从数据包的其它数据中下载 stock_st/，训练特征工程需要它来过滤每日 ST 股票。"
        )

    file_list = sorted(glob.glob(os.path.join(stock_st_dir, "*.csv")))
    if len(file_list) == 0:
        raise FileNotFoundError(f"stock_st 目录中没有 CSV 文件: {stock_st_dir}")

    by_date: dict[int, set[str]] = {}

    for file_path in file_list:
        try:
            st_df = pd.read_csv(file_path, usecols=lambda c: c in {"ts_code", "trade_date"})
        except Exception as e:
            print(f"读取 stock_st 失败: {file_path}, 错误: {e}")
            continue

        missing = {"ts_code", "trade_date"} - set(st_df.columns)
        if missing:
            print(f"跳过 stock_st 文件，缺少列 {missing}: {file_path}")
            continue

        st_df = st_df.dropna(subset=["ts_code", "trade_date"]).copy()
        st_df["trade_date"] = pd.to_numeric(st_df["trade_date"], errors="coerce")
        st_df = st_df.dropna(subset=["trade_date"])
        st_df["trade_date"] = st_df["trade_date"].astype("int32")
        st_df["ts_code"] = st_df["ts_code"].astype(str)

        for trade_date, g in st_df.groupby("trade_date"):
            by_date.setdefault(int(trade_date), set()).update(g["ts_code"].tolist())

    if not by_date:
        raise ValueError(f"没有从 stock_st 目录读取到有效 ST 列表: {stock_st_dir}")

    dates = sorted(by_date)
    print(
        f"已读取每日 ST 列表: dates={len(dates)}, "
        f"range={dates[0]}-{dates[-1]}"
    )
    return by_date, dates


def latest_st_codes_for_date(
    trade_date: int,
    stock_st_by_date: dict[int, set[str]],
    stock_st_dates: list[int],
) -> set[str]:
    idx = bisect_right(stock_st_dates, int(trade_date)) - 1
    if idx < 0:
        return set()
    return stock_st_by_date.get(stock_st_dates[idx], set())


# ============================================================
# 2. 一次性读取所有 daily CSV
# ============================================================

def load_all_daily(
    daily_dir: str,
    valid_codes: set,
    stock_st_by_date: dict[int, set[str]],
    stock_st_dates: list[int],
    usecols=None
) -> pd.DataFrame:
    """
    正确读取方式：
    不要每只股票重复遍历所有 daily 文件。
    而是所有 daily 文件只读一次，然后合并成全市场大表。
    """

    if usecols is None:
        usecols = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount"
        ]

    file_list = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))

    if len(file_list) == 0:
        raise FileNotFoundError(f"没有在目录中找到 CSV 文件: {daily_dir}")

    all_dfs = []

    for i, file_path in enumerate(file_list):
        print(f"[{i + 1}/{len(file_list)}] 正在读取: {file_path}")

        try:
            # 先读表头，避免 usecols 缺失时报错
            header = pd.read_csv(file_path, nrows=0)
            available_cols = [c for c in usecols if c in header.columns]

            if "ts_code" not in available_cols:
                print(f"  -> 跳过，缺少 ts_code: {file_path}")
                continue

            df_day = pd.read_csv(file_path, usecols=available_cols)

            # 过滤股票池
            df_day = df_day[df_day["ts_code"].isin(valid_codes)]

            if "trade_date" in df_day.columns:
                day_dates = pd.to_numeric(df_day["trade_date"], errors="coerce").dropna().astype("int32")
                for trade_date in day_dates.unique():
                    st_codes = latest_st_codes_for_date(trade_date, stock_st_by_date, stock_st_dates)
                    if st_codes:
                        mask = (df_day["trade_date"].astype(str) == str(int(trade_date))) & df_day["ts_code"].isin(st_codes)
                        df_day = df_day[~mask]

            if df_day.empty:
                continue

            all_dfs.append(df_day)

        except Exception as e:
            print(f"读取失败: {file_path}, 错误: {e}")

    if len(all_dfs) == 0:
        raise ValueError("没有读到任何有效日频数据。")

    df = pd.concat(all_dfs, ignore_index=True)

    # 基础缺失过滤
    required_cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount"
    ]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # 类型优化，减少内存
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount"
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce").astype("int32")

    df = df.dropna()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    return df


# ============================================================
# 3. 特征工程
# ============================================================

def add_features_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    对全市场数据一次性做特征工程：
    - 横截面特征：按 trade_date 排名
    - 时间序列特征：按 ts_code rolling / pct_change / shift
    """

    df = df.copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 3.1 横截面排名特征
    # ------------------------------------------------------------

    print("正在计算横截面排名特征...")

    df["close_rank"] = df.groupby("trade_date")["close"].rank(pct=True).astype("float32")
    df["vol_rank"] = df.groupby("trade_date")["vol"].rank(pct=True).astype("float32")
    df["amount_rank"] = df.groupby("trade_date")["amount"].rank(pct=True).astype("float32")

    # ------------------------------------------------------------
    # 3.2 时间序列特征
    # ------------------------------------------------------------

    print("正在计算时间序列特征...")

    g = df.groupby("ts_code", group_keys=False)

    # 如果没有 pre_close，就用前一日 close 近似
    if "pre_close" not in df.columns:
        df["pre_close"] = g["close"].shift(1)

    df["pre_close"] = df["pre_close"].replace(0, np.nan)

    # K线结构
    df["amplitude"] = ((df["high"] - df["low"]) / df["pre_close"]).astype("float32")

    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / high_low).astype("float32")

    # 收益率
    df["ret_1"] = g["close"].pct_change(1).astype("float32")
    df["ret_5"] = g["close"].pct_change(5).astype("float32")
    df["ret_10"] = g["close"].pct_change(10).astype("float32")
    df["ret_20"] = g["close"].pct_change(20).astype("float32")

    # 均线
    df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean()).astype("float32")
    df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean()).astype("float32")
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean()).astype("float32")

    df["rel_ma5"] = (df["close"] / df["ma5"] - 1).astype("float32")
    df["rel_ma10"] = (df["close"] / df["ma10"] - 1).astype("float32")
    df["rel_ma20"] = (df["close"] / df["ma20"] - 1).astype("float32")

    # RSI 14
    print("正在计算 RSI...")

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain_mean = gain.groupby(df["ts_code"]).transform(
        lambda x: x.rolling(14).mean()
    )
    loss_mean = loss.groupby(df["ts_code"]).transform(
        lambda x: x.rolling(14).mean()
    )

    rs = gain_mean / loss_mean.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).astype("float32")

    # MACD
    print("正在计算 MACD...")

    ema12 = g["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema26 = g["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )

    df["macd_dif"] = (ema12 - ema26).astype("float32")
    df["macd_dea"] = df.groupby("ts_code")["macd_dif"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    ).astype("float32")
    df["macd_bar"] = (df["macd_dif"] - df["macd_dea"]).astype("float32")

    # 波动率
    df["vol_std_10"] = g["ret_1"].transform(lambda x: x.rolling(10).std()).astype("float32")
    df["vol_std_20"] = g["ret_1"].transform(lambda x: x.rolling(20).std()).astype("float32")

    # 成交量 / 成交额均线
    df["vol_ma5"] = g["vol"].transform(lambda x: x.rolling(5).mean()).astype("float32")
    df["vol_ma20"] = g["vol"].transform(lambda x: x.rolling(20).mean()).astype("float32")

    df["amount_ma5"] = g["amount"].transform(lambda x: x.rolling(5).mean()).astype("float32")
    df["amount_ma20"] = g["amount"].transform(lambda x: x.rolling(20).mean()).astype("float32")

    # 量价变化
    df["vol_chg"] = g["vol"].pct_change(1).astype("float32")
    df["amount_chg"] = g["amount"].pct_change(1).astype("float32")

    # ------------------------------------------------------------
    # 3.3 标签
    # ------------------------------------------------------------

    print("正在构造标签...")

    df["label_1d"] = (g["close"].shift(-1) / df["close"] - 1).astype("float32")
    df["label_2d"] = (g["close"].shift(-2) / df["close"] - 1).astype("float32")
    df["label_5d"] = (g["close"].shift(-5) / df["close"] - 1).astype("float32")
    df["label_10d"] = (g["close"].shift(-10) / df["close"] - 1).astype("float32")

    df["label_direction_1d"] = (df["label_1d"] > 0).astype("int8")
    df["next_close"] = g["close"].shift(-1).astype("float32")

    # 清理无穷值
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# 4. 安全保存函数
# ============================================================

def safe_save_features(df: pd.DataFrame):
    """
    优先保存 parquet。
    如果没有 pyarrow / fastparquet，则自动保存为 pickle。
    同时可以按需保存 CSV。
    """

    print("准备保存特征数据...")

    # 方案 1：优先保存 parquet
    try:
        df.to_parquet(FEATURE_PARQUET_PATH, index=False)
        print(f"已保存 parquet: {FEATURE_PARQUET_PATH}")
        return FEATURE_PARQUET_PATH

    except ImportError as e:
        print("当前环境缺少 pyarrow 或 fastparquet，无法保存 parquet。")
        print("将自动改为保存 pickle。")
        print(f"parquet 报错信息: {e}")

    except Exception as e:
        print("保存 parquet 失败，将自动改为保存 pickle。")
        print(f"parquet 报错信息: {e}")

    # 方案 2：保存 pickle
    try:
        df.to_pickle(FEATURE_PICKLE_PATH)
        print(f"已保存 pickle: {FEATURE_PICKLE_PATH}")
        print("之后可以用 pd.read_pickle() 读取。")
        return FEATURE_PICKLE_PATH

    except Exception as e:
        print("保存 pickle 失败。")
        print(f"pickle 报错信息: {e}")

    # 方案 3：最后 fallback 到 csv
    try:
        df.to_csv(FEATURE_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"已保存 csv: {FEATURE_CSV_PATH}")
        return FEATURE_CSV_PATH

    except Exception as e:
        print("保存 csv 也失败。")
        print(f"csv 报错信息: {e}")
        raise RuntimeError("所有保存方式均失败，请检查磁盘空间和权限。")


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("开始股票特征工程")
    print("=" * 80)

    # ------------------------------------------------------------
    # 5.1 读取 basic
    # ------------------------------------------------------------

    print("正在读取 basic.csv...")

    if not os.path.exists(BASIC_PATH):
        raise FileNotFoundError(f"找不到 basic.csv: {BASIC_PATH}")

    basic_df = pd.read_csv(BASIC_PATH)

    print("原始股票数量:", len(basic_df))

    basic_filtered = filter_stock(basic_df)

    print("过滤后股票数量:", len(basic_filtered))

    basic_filtered.to_csv(
        BASIC_FILTERED_PATH,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"已保存过滤股票池: {BASIC_FILTERED_PATH}")

    valid_codes = set(basic_filtered["ts_code"].tolist())

    # ------------------------------------------------------------
    # 5.2 读取每日 stock_st
    # ------------------------------------------------------------

    print("=" * 80)
    print("开始读取每日 ST 股票列表")
    print("=" * 80)

    stock_st_by_date, stock_st_dates = load_stock_st_by_date(STOCK_ST_DIR)

    # ------------------------------------------------------------
    # 5.3 读取所有 daily
    # ------------------------------------------------------------

    print("=" * 80)
    print("开始读取全市场日频数据")
    print("=" * 80)

    df_daily = load_all_daily(
        daily_dir=DAILY_DIR,
        valid_codes=valid_codes,
        stock_st_by_date=stock_st_by_date,
        stock_st_dates=stock_st_dates,
    )

    print("全市场数据 shape:", df_daily.shape)
    print(df_daily.head())

    # ------------------------------------------------------------
    # 5.4 特征工程
    # ------------------------------------------------------------

    print("=" * 80)
    print("开始计算特征")
    print("=" * 80)

    df_feat = add_features_all(df_daily)

    del df_daily
    gc.collect()

    # 删除特征和标签缺失行
    df_feat = df_feat.dropna().reset_index(drop=True)

    print("特征数据 shape:", df_feat.shape)
    print("特征列数量:", len(df_feat.columns))
    print(df_feat.head())

    # ------------------------------------------------------------
    # 5.5 保存
    # ------------------------------------------------------------

    print("=" * 80)
    print("开始保存")
    print("=" * 80)

    saved_path = safe_save_features(df_feat)

    print("=" * 80)
    print("全部完成")
    print(f"最终保存路径: {saved_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
