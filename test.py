import pandas as pd
import numpy as np

# df_feat = pd.read_pickle("./Datasets/processed/all_stock_features.pkl")
# 如果你保存的是 parquet，就改成：
df_feat = pd.read_parquet("Datasets/processed/all_stock_features.parquet")

print("shape:", df_feat.shape)
print("date range:", df_feat["trade_date"].min(), df_feat["trade_date"].max())
print("stock num:", df_feat["ts_code"].nunique())

print("\n每个日期股票数量：")
print(df_feat.groupby("trade_date")["ts_code"].nunique().describe())

print("\n每只股票样本数量：")
print(df_feat.groupby("ts_code")["trade_date"].nunique().describe())

print("\nlabel 分布：")
for col in ["label_1d", "label_2d", "label_5d", "label_10d"]:
    print("=" * 50)
    print(col)
    print(df_feat[col].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

print("\n是否有 inf：")
numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
print(np.isinf(df_feat[numeric_cols]).sum().sort_values(ascending=False).head(10))

print("\n是否有 NaN：")
print(df_feat.isna().sum().sort_values(ascending=False).head(10))
