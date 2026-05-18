import os
import pandas as pd
import numpy as np


# ============================================================
# 路径配置：默认全部使用 parquet
# ============================================================

DATA_PATH = "Datasets/processed/all_stock_features.parquet"
SAVE_PATH = "Datasets/processed/all_stock_features_with_oc_label.parquet"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"找不到输入文件: {DATA_PATH}\n"
            f"请确认你的特征文件是否保存在该路径。\n"
            f"如果文件名不同，请修改 DATA_PATH。"
        )

    print("=" * 80)
    print("读取 Parquet 特征文件")
    print("=" * 80)
    print("DATA_PATH:", DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    print("原始数据 shape:", df.shape)
    print("日期范围:", df["trade_date"].min(), "-", df["trade_date"].max())
    print("股票数量:", df["ts_code"].nunique())

    # 确保按股票和日期排序
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    g = df.groupby("ts_code", group_keys=False)

    # ============================================================
    # 构造 next-day open-to-close 标签
    # 含义：
    # t 日收盘后生成信号
    # t+1 日开盘买入
    # t+1 日收盘计算收益
    # label_oc_1d = close[t+1] / open[t+1] - 1
    # ============================================================

    df["next_open"] = g["open"].shift(-1)
    df["next_close_real"] = g["close"].shift(-1)

    df["label_oc_1d"] = df["next_close_real"] / df["next_open"] - 1

    # 清理异常
    df = df.replace([np.inf, -np.inf], np.nan)

    print("=" * 80)
    print("label_oc_1d 分布，dropna 前")
    print("=" * 80)
    print(df["label_oc_1d"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    before_shape = df.shape

    # 只删除 label_oc_1d 缺失的最后一天样本
    df = df.dropna(subset=["label_oc_1d"]).reset_index(drop=True)

    print("=" * 80)
    print("处理完成")
    print("=" * 80)
    print("before shape:", before_shape)
    print("after shape:", df.shape)

    print("=" * 80)
    print("label_oc_1d 分布，dropna 后")
    print("=" * 80)
    print(df["label_oc_1d"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    df.to_parquet(SAVE_PATH, index=False)

    print("=" * 80)
    print("保存完成")
    print("=" * 80)
    print("SAVE_PATH:", SAVE_PATH)


if __name__ == "__main__":
    main()