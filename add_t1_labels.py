import os
import numpy as np
import pandas as pd

INPUT_PATH = "Datasets/processed/all_stock_features_with_oc_label.parquet"
OUTPUT_PATH = "Datasets/processed/all_stock_features_with_t1_labels.parquet"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    print("=" * 80)
    print("读取数据")
    print("=" * 80)

    df = pd.read_parquet(INPUT_PATH)

    print("原始 shape:", df.shape)
    print("date range:", df["trade_date"].min(), "-", df["trade_date"].max())
    print("stock num:", df["ts_code"].nunique())

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    g = df.groupby("ts_code", group_keys=False)

    # t+1 open
    df["next_open"] = g["open"].shift(-1)

    # t+2 open
    df["next2_open"] = g["open"].shift(-2)

    # t+2 close，可选备用
    df["next2_close"] = g["close"].shift(-2)

    # 严格 T+1：
    # t 日收盘生成信号，t+1 日开盘买入，t+2 日开盘卖出
    df["label_oo_1d"] = df["next2_open"] / df["next_open"] - 1

    # 可选：
    # t+1 开盘买入，t+2 收盘估值
    df["label_oc_2d"] = df["next2_close"] / df["next_open"] - 1

    df = df.replace([np.inf, -np.inf], np.nan)

    print("=" * 80)
    print("标签分布")
    print("=" * 80)

    for col in ["label_oc_1d", "label_oo_1d", "label_oc_2d"]:
        if col in df.columns:
            print("-" * 80)
            print(col)
            print(df[col].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    before = df.shape
    df = df.dropna(subset=["label_oo_1d", "label_oc_2d"]).reset_index(drop=True)

    print("=" * 80)
    print("处理完成")
    print("=" * 80)
    print("before:", before)
    print("after:", df.shape)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
