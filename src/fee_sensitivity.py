import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def calc_metrics(nav_df):
    nav = nav_df["nav"].astype(float)
    ret = nav_df["net_ret"].astype(float)

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav_df)) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
    max_drawdown = (nav / nav.cummax() - 1).min()
    win_rate = (ret > 0).mean()
    avg_turnover = nav_df["turnover"].mean()

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "avg_turnover": float(avg_turnover),
        "num_days": int(len(nav_df)),
    }


def recompute_nav(nav_path, fee_rate):
    df = pd.read_csv(nav_path)

    need_cols = ["trade_date", "portfolio_ret", "turnover"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"{nav_path} 缺少列 {c}，实际列为: {df.columns.tolist()}")

    df = df.sort_values("trade_date").reset_index(drop=True).copy()

    df["cost"] = df["turnover"] * fee_rate
    df["net_ret"] = df["portfolio_ret"] - df["cost"]
    df["nav"] = (1.0 + df["net_ret"]).cumprod()

    return df


def plot_sensitivity(nav_dict, out_dir, prefix):
    plt.figure(figsize=(11, 6))

    for label, df in nav_dict.items():
        plt.plot(df["trade_date"].astype(str), df["nav"], label=label)

    first = next(iter(nav_dict.values()))
    x = first["trade_date"].astype(str)
    plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title(f"Fee Sensitivity NAV - {prefix}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_fee_sensitivity_nav.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))

    for label, df in nav_dict.items():
        nav = df["nav"].astype(float)
        dd = nav / nav.cummax() - 1
        plt.plot(df["trade_date"].astype(str), dd, label=label)

    plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title(f"Fee Sensitivity Drawdown - {prefix}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_fee_sensitivity_drawdown.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav_paths", nargs="+", required=True)
    parser.add_argument("--strategy_names", nargs="+", required=True)
    parser.add_argument("--fee_rates", nargs="+", type=float, default=[0.0003, 0.0005, 0.001, 0.002])
    parser.add_argument("--out_dir", type=str, default="outputs_benchmark/fee_sensitivity")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    if len(args.nav_paths) != len(args.strategy_names):
        raise ValueError("nav_paths 和 strategy_names 数量必须一致")

    all_metrics = []

    for nav_path, strategy_name in zip(args.nav_paths, args.strategy_names):
        nav_dict = {}

        for fee in args.fee_rates:
            df = recompute_nav(nav_path, fee)
            label = f"fee={fee}"

            nav_dict[label] = df

            out_nav_path = os.path.join(
                args.out_dir,
                f"{strategy_name}_fee_{fee:.4f}_nav.csv".replace(".", "p"),
            )
            df.to_csv(out_nav_path, index=False, encoding="utf-8-sig")

            m = calc_metrics(df)
            m["strategy"] = strategy_name
            m["fee_rate"] = fee
            all_metrics.append(m)

        plot_sensitivity(nav_dict, args.out_dir, strategy_name)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[
        [
            "strategy",
            "fee_rate",
            "total_return",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "avg_turnover",
            "num_days",
        ]
    ]

    metrics_df.to_csv(
        os.path.join(args.out_dir, "fee_sensitivity_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Fee sensitivity metrics")
    print("=" * 80)
    print(metrics_df)
    print("=" * 80)
    print("输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
