import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_label_path(path: str):
    if os.path.exists(path):
        return path
    candidates = [
        "Datasets/processed/all_stock_features_with_t1_labels.parquet",
        "../Datasets/processed/all_stock_features_with_t1_labels.parquet",
        "Datasets/processed/all_stock_features_with_oc_label.parquet",
        "../Datasets/processed/all_stock_features_with_oc_label.parquet",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到数据文件: {path}；候选路径也不存在: {candidates}")


def safe_corr(x: pd.Series, y: pd.Series, method: str):
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    if x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    return x[valid].corr(y[valid], method=method)


def calc_group_corr(df: pd.DataFrame, group_col: str, label_a: str, label_b: str):
    rows = []
    for k, g in df.groupby(group_col):
        rows.append({
            group_col: k,
            "num_samples": int(len(g)),
            "pearson": safe_corr(g[label_a], g[label_b], method="pearson"),
            "spearman": safe_corr(g[label_a], g[label_b], method="spearman"),
            f"{label_a}_mean": float(g[label_a].mean()),
            f"{label_b}_mean": float(g[label_b].mean()),
            f"{label_a}_std": float(g[label_a].std()),
            f"{label_b}_std": float(g[label_b].std()),
        })
    return pd.DataFrame(rows)


def plot_monthly_corr(monthly: pd.DataFrame, out_dir: str):
    x = monthly["month"].astype(str)
    plt.figure(figsize=(11, 5))
    plt.plot(x, monthly["pearson"], marker="o", label="Pearson")
    plt.plot(x, monthly["spearman"], marker="o", label="Spearman")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Correlation")
    plt.title("Monthly Correlation: label_oc_1d vs label_oo_1d")
    plt.xticks(x[::max(len(x)//12, 1)], rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oc_oo_monthly_correlation.png"), dpi=200)
    plt.close()


def plot_yearly_corr(yearly: pd.DataFrame, out_dir: str):
    x = yearly["year"].astype(str)
    idx = np.arange(len(yearly))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(idx - width / 2, yearly["pearson"], width=width, label="Pearson")
    plt.bar(idx + width / 2, yearly["spearman"], width=width, label="Spearman")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Correlation")
    plt.title("Yearly Correlation: label_oc_1d vs label_oo_1d")
    plt.xticks(idx, x)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oc_oo_yearly_correlation.png"), dpi=200)
    plt.close()


def plot_hexbin(df: pd.DataFrame, label_a: str, label_b: str, out_dir: str, max_points: int = 500000):
    data = df[[label_a, label_b]].dropna().copy()
    q_low_a, q_high_a = data[label_a].quantile([0.005, 0.995])
    q_low_b, q_high_b = data[label_b].quantile([0.005, 0.995])
    data = data[
        (data[label_a] >= q_low_a) & (data[label_a] <= q_high_a) &
        (data[label_b] >= q_low_b) & (data[label_b] <= q_high_b)
    ]
    if len(data) > max_points:
        data = data.sample(max_points, random_state=42)

    plt.figure(figsize=(7, 6))
    plt.hexbin(data[label_a], data[label_b], gridsize=80, bins="log")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(label_a)
    plt.ylabel(label_b)
    plt.title("Hexbin: label_oc_1d vs label_oo_1d")
    cb = plt.colorbar()
    cb.set_label("log(count)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oc_oo_hexbin.png"), dpi=200)
    plt.close()


def plot_distribution(df: pd.DataFrame, label_a: str, label_b: str, out_dir: str):
    data = df[[label_a, label_b]].dropna().copy()
    q_low = min(data[label_a].quantile(0.005), data[label_b].quantile(0.005))
    q_high = max(data[label_a].quantile(0.995), data[label_b].quantile(0.995))

    plt.figure(figsize=(9, 5))
    plt.hist(data[label_a].clip(q_low, q_high), bins=120, alpha=0.6, label=label_a, density=True)
    plt.hist(data[label_b].clip(q_low, q_high), bins=120, alpha=0.6, label=label_b, density=True)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.title("Label Distribution Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "oc_oo_distribution.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Datasets/processed/all_stock_features_with_t1_labels.parquet")
    parser.add_argument("--out_dir", type=str, default="outputs_benchmark/label_correlation")
    parser.add_argument("--label_a", type=str, default="label_oc_1d")
    parser.add_argument("--label_b", type=str, default="label_oo_1d")
    parser.add_argument("--start_date", type=int, default=0)
    parser.add_argument("--end_date", type=int, default=99999999)
    parser.add_argument("--daily_sample", type=int, default=0, help="0 表示每日横截面全量计算；如果太慢可设为 2000000")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    data_path = find_label_path(args.data_path)

    print("=" * 80)
    print("读取数据")
    print("=" * 80)
    print("data_path:", data_path)

    cols = ["ts_code", "trade_date", args.label_a, args.label_b]
    df = pd.read_parquet(data_path, columns=cols)

    df = df[(df["trade_date"] >= args.start_date) & (df["trade_date"] <= args.end_date)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[args.label_a, args.label_b]).copy()

    df["year"] = (df["trade_date"] // 10000).astype(int)
    df["month"] = (df["trade_date"] // 100).astype(int)

    print("shape:", df.shape)
    print("date range:", df["trade_date"].min(), "-", df["trade_date"].max())
    print("stock num:", df["ts_code"].nunique())

    overall = {
        "scope": "all",
        "num_samples": int(len(df)),
        "num_days": int(df["trade_date"].nunique()),
        "num_stocks": int(df["ts_code"].nunique()),
        "pearson": safe_corr(df[args.label_a], df[args.label_b], method="pearson"),
        "spearman": safe_corr(df[args.label_a], df[args.label_b], method="spearman"),
        f"{args.label_a}_mean": float(df[args.label_a].mean()),
        f"{args.label_b}_mean": float(df[args.label_b].mean()),
        f"{args.label_a}_std": float(df[args.label_a].std()),
        f"{args.label_b}_std": float(df[args.label_b].std()),
        f"{args.label_a}_p01": float(df[args.label_a].quantile(0.01)),
        f"{args.label_b}_p01": float(df[args.label_b].quantile(0.01)),
        f"{args.label_a}_p50": float(df[args.label_a].quantile(0.50)),
        f"{args.label_b}_p50": float(df[args.label_b].quantile(0.50)),
        f"{args.label_a}_p99": float(df[args.label_a].quantile(0.99)),
        f"{args.label_b}_p99": float(df[args.label_b].quantile(0.99)),
    }
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(os.path.join(args.out_dir, "oc_oo_overall_correlation.csv"), index=False, encoding="utf-8-sig")

    yearly = calc_group_corr(df, "year", args.label_a, args.label_b)
    yearly.to_csv(os.path.join(args.out_dir, "oc_oo_yearly_correlation.csv"), index=False, encoding="utf-8-sig")

    monthly = calc_group_corr(df, "month", args.label_a, args.label_b)
    monthly.to_csv(os.path.join(args.out_dir, "oc_oo_monthly_correlation.csv"), index=False, encoding="utf-8-sig")

    daily_df = df
    if args.daily_sample and len(df) > args.daily_sample:
        daily_df = df.sample(args.daily_sample, random_state=42)
    daily = calc_group_corr(daily_df, "trade_date", args.label_a, args.label_b)
    daily.to_csv(os.path.join(args.out_dir, "oc_oo_daily_correlation.csv"), index=False, encoding="utf-8-sig")

    daily_summary = pd.DataFrame([{
        "daily_pearson_mean": float(daily["pearson"].mean()),
        "daily_pearson_std": float(daily["pearson"].std()),
        "daily_spearman_mean": float(daily["spearman"].mean()),
        "daily_spearman_std": float(daily["spearman"].std()),
        "num_days": int(daily["trade_date"].nunique()),
    }])
    daily_summary.to_csv(os.path.join(args.out_dir, "oc_oo_daily_correlation_summary.csv"), index=False, encoding="utf-8-sig")

    plot_monthly_corr(monthly, args.out_dir)
    plot_yearly_corr(yearly, args.out_dir)
    plot_hexbin(df, args.label_a, args.label_b, args.out_dir)
    plot_distribution(df, args.label_a, args.label_b, args.out_dir)

    print("=" * 80)
    print("整体相关性")
    print("=" * 80)
    print(overall_df)

    print("=" * 80)
    print("年度相关性")
    print("=" * 80)
    print(yearly)

    print("=" * 80)
    print("最近 12 个月相关性")
    print("=" * 80)
    print(monthly.tail(12))

    print("=" * 80)
    print("每日横截面相关性 summary")
    print("=" * 80)
    print(daily_summary)

    print("=" * 80)
    print("完成，输出目录:", args.out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
