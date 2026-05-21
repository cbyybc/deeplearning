import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INDEX_FILES = {
    "000001.SH": "上证指数",
    "000300.SH": "沪深300",
    "399006.SZ": "创业板指",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_columns(df):
    """
    兼容不同 CSV 列名：
    1. Tushare: trade_date, open, close
    2. 中文: 日期, 开盘, 收盘
    3. 英文大小写混合: Date, Open, Close
    """
    rename = {}

    for c in df.columns:
        c_str = str(c).strip()
        c_low = c_str.lower()

        if c_str in ["日期", "交易日期"] or c_low in ["date", "trade_date", "datetime"]:
            rename[c] = "trade_date"
        elif c_str in ["开盘", "开盘价"] or c_low in ["open", "open_price"]:
            rename[c] = "open"
        elif c_str in ["收盘", "收盘价"] or c_low in ["close", "close_price"]:
            rename[c] = "close"
        elif c_str in ["最高", "最高价"] or c_low in ["high"]:
            rename[c] = "high"
        elif c_str in ["最低", "最低价"] or c_low in ["low"]:
            rename[c] = "low"

    df = df.rename(columns=rename)

    need_cols = ["trade_date", "open", "close"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}\n"
            f"当前列名: {df.columns.tolist()}\n"
            f"需要至少包含 trade_date/open/close 或 日期/开盘/收盘"
        )

    df = df[["trade_date", "open", "close"]].copy()

    # 兼容 20250102、2025-01-02、2025/01/02
    raw_date = df["trade_date"].astype(str).str.strip()

    if raw_date.str.match(r"^\d{8}$").all():
        df["trade_date"] = raw_date.astype(int)
    else:
        df["trade_date"] = pd.to_datetime(raw_date, errors="coerce").dt.strftime("%Y%m%d").astype("float")

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["trade_date", "open", "close"]).copy()
    df["trade_date"] = df["trade_date"].astype(int)

    df = df.sort_values("trade_date").reset_index(drop=True)

    return df


def calc_nav_metrics(df):
    df = df.copy()
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 指数基准默认按 close-to-close 计算每日收益
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["nav"] = (1.0 + df["ret"]).cumprod()

    ret = df["ret"].astype(float)
    nav = df["nav"].astype(float)

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(df)) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
    max_drawdown = (nav / nav.cummax() - 1).min()
    win_rate = (ret > 0).mean()

    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_days": int(len(df)),
    }

    return metrics, df


def plot_benchmark_nav(nav_dict, out_dir):
    plt.figure(figsize=(11, 6))

    for name, df in nav_dict.items():
        plt.plot(df["trade_date"].astype(str), df["nav"], label=name)

    first = next(iter(nav_dict.values()))
    x = first["trade_date"].astype(str)
    plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Market Index Benchmark NAV")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "index_benchmark_nav.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))

    for name, df in nav_dict.items():
        nav = df["nav"].astype(float)
        dd = nav / nav.cummax() - 1
        plt.plot(df["trade_date"].astype(str), dd, label=name)

    plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Market Index Benchmark Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "index_benchmark_drawdown.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market_dir", type=str, default="../Datasets/market")
    parser.add_argument("--out_dir", type=str, default="outputs_benchmark/index")
    parser.add_argument("--start_date", type=int, default=20250101)
    parser.add_argument("--end_date", type=int, default=20251231)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    all_metrics = []
    nav_dict = {}

    for code, name in INDEX_FILES.items():
        path = os.path.join(args.market_dir, f"{code}.csv")

        if not os.path.exists(path):
            print(f"[WARN] 找不到文件，跳过: {path}")
            continue

        print("=" * 80)
        print(f"读取指数: {code} {name}")
        print(path)

        raw = pd.read_csv(path)
        df = normalize_columns(raw)

        df = df[
            (df["trade_date"] >= args.start_date) &
            (df["trade_date"] <= args.end_date)
        ].copy()

        if df.empty:
            print(f"[WARN] {code} 在指定日期范围内没有数据，跳过")
            continue

        metrics, nav_df = calc_nav_metrics(df)

        metrics["index_code"] = code
        metrics["index_name"] = name

        all_metrics.append(metrics)
        nav_dict[name] = nav_df

        nav_df.to_csv(
            os.path.join(args.out_dir, f"{code}_nav.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"{name} metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    if not all_metrics:
        raise RuntimeError("没有成功读取任何指数数据，请检查 /dataset/market 下的 CSV 文件和列名")

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df = metrics_df[
        [
            "index_code",
            "index_name",
            "total_return",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "num_days",
        ]
    ]

    metrics_df.to_csv(
        os.path.join(args.out_dir, "index_benchmark_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_benchmark_nav(nav_dict, args.out_dir)

    print("=" * 80)
    print("Market index benchmark summary")
    print("=" * 80)
    print(metrics_df)
    print("=" * 80)
    print("输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
