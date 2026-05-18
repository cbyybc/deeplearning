import os
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_daily_ic(pred_df, pred_col="pred", label_col="label"):
    def _corr(x):
        if x[pred_col].nunique() <= 1 or x[label_col].nunique() <= 1:
            return np.nan
        return x[pred_col].corr(x[label_col])

    return pred_df.groupby("trade_date").apply(_corr).dropna()


def calc_daily_rank_ic(pred_df, pred_col="pred", label_col="label"):
    def _corr(x):
        if x[pred_col].nunique() <= 1 or x[label_col].nunique() <= 1:
            return np.nan
        return x[pred_col].corr(x[label_col], method="spearman")

    return pred_df.groupby("trade_date").apply(_corr).dropna()


def plot_training_loss(output_dir, fig_dir):
    path = os.path.join(output_dir, "training_history.csv")
    if not os.path.exists(path):
        print(f"跳过 loss 曲线，找不到: {path}")
        return

    df = pd.read_csv(path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "viz_loss_curve.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"已保存: {save_path}")


def plot_daily_ic(pred_df, fig_dir, prefix):
    daily_ic = calc_daily_ic(pred_df)
    daily_rank_ic = calc_daily_rank_ic(pred_df)

    # IC 时间序列
    plt.figure(figsize=(10, 5))
    plt.plot(daily_ic.index.astype(str), daily_ic.values, label="Daily IC")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axhline(daily_ic.mean(), linestyle="--", linewidth=1, label=f"Mean IC={daily_ic.mean():.4f}")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.title(f"{prefix} Daily IC")
    plt.xticks(daily_ic.index.astype(str)[::max(len(daily_ic)//10, 1)], rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_daily_ic.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    # RankIC 时间序列
    plt.figure(figsize=(10, 5))
    plt.plot(daily_rank_ic.index.astype(str), daily_rank_ic.values, label="Daily Rank IC")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axhline(daily_rank_ic.mean(), linestyle="--", linewidth=1, label=f"Mean RankIC={daily_rank_ic.mean():.4f}")
    plt.xlabel("Date")
    plt.ylabel("Rank IC")
    plt.title(f"{prefix} Daily Rank IC")
    plt.xticks(daily_rank_ic.index.astype(str)[::max(len(daily_rank_ic)//10, 1)], rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_daily_rank_ic.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    # IC 分布
    plt.figure(figsize=(8, 5))
    plt.hist(daily_ic.values, bins=40, alpha=0.8)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axvline(daily_ic.mean(), linestyle="--", linewidth=1, label=f"Mean={daily_ic.mean():.4f}")
    plt.xlabel("Daily IC")
    plt.ylabel("Count")
    plt.title(f"{prefix} IC Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_ic_distribution.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")


def plot_prediction_distribution(pred_df, fig_dir, prefix):
    # 预测分数分布
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["pred"].values, bins=80, alpha=0.8)
    plt.xlabel("Predicted Score")
    plt.ylabel("Count")
    plt.title(f"{prefix} Prediction Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_pred_distribution.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    # label 分布
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["label"].values, bins=80, alpha=0.8)
    plt.xlabel("True Label")
    plt.ylabel("Count")
    plt.title(f"{prefix} Label Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_label_distribution.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")


def plot_decile_return(pred_df, fig_dir, prefix):
    """
    每天按 pred 分成 10 组，计算每组平均真实收益。
    如果模型有效，通常高分组平均收益应该高于低分组。
    """
    rows = []

    for date, g in pred_df.groupby("trade_date"):
        g = g.copy()
        if len(g) < 50:
            continue

        try:
            g["decile"] = pd.qcut(g["pred"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        tmp = g.groupby("decile")["label"].mean().reset_index()
        tmp["trade_date"] = date
        rows.append(tmp)

    if len(rows) == 0:
        print(f"跳过 decile return，数据不足: {prefix}")
        return

    decile_df = pd.concat(rows, ignore_index=True)
    mean_ret = decile_df.groupby("decile")["label"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(mean_ret.index.astype(str), mean_ret.values)
    plt.xlabel("Prediction Decile, 0=Lowest, 9=Highest")
    plt.ylabel("Average Future Return")
    plt.title(f"{prefix} Decile Average Return")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"{prefix.lower()}_decile_return.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    save_csv = os.path.join(fig_dir, f"{prefix.lower()}_decile_return.csv")
    mean_ret.to_csv(save_csv, header=["mean_label"])
    print(f"已保存: {save_csv}")


def plot_nav_and_drawdown(output_dir, fig_dir):
    path = os.path.join(output_dir, "backtest", "nav_curve.csv")
    if not os.path.exists(path):
        print(f"跳过净值曲线，找不到: {path}")
        return

    nav_df = pd.read_csv(path)
    nav_df["trade_date"] = nav_df["trade_date"].astype(str)

    nav = nav_df["nav"].astype(float)
    drawdown = nav / nav.cummax() - 1

    # 净值曲线
    plt.figure(figsize=(10, 5))
    plt.plot(nav_df["trade_date"], nav)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Backtest NAV Curve")
    plt.xticks(nav_df["trade_date"][::max(len(nav_df)//10, 1)], rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "viz_backtest_nav.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    # 回撤曲线
    plt.figure(figsize=(10, 5))
    plt.plot(nav_df["trade_date"], drawdown)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Backtest Drawdown Curve")
    plt.xticks(nav_df["trade_date"][::max(len(nav_df)//10, 1)], rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "viz_backtest_drawdown.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")

    # 每日收益分布
    daily_ret = nav.pct_change().fillna(0)
    plt.figure(figsize=(8, 5))
    plt.hist(daily_ret.values, bins=50, alpha=0.8)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Daily Portfolio Return")
    plt.ylabel("Count")
    plt.title("Backtest Daily Return Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "viz_backtest_daily_return_distribution.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")


def plot_turnover(output_dir, fig_dir):
    path = os.path.join(output_dir, "backtest", "nav_curve.csv")
    if not os.path.exists(path):
        return

    nav_df = pd.read_csv(path)
    if "turnover" not in nav_df.columns:
        print("nav_curve.csv 中没有 turnover 列，跳过换手率图。")
        return

    nav_df["trade_date"] = nav_df["trade_date"].astype(str)

    plt.figure(figsize=(10, 5))
    plt.plot(nav_df["trade_date"], nav_df["turnover"])
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.title("Daily Turnover")
    plt.xticks(nav_df["trade_date"][::max(len(nav_df)//10, 1)], rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "viz_daily_turnover.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存: {save_path}")


def print_metrics(output_dir):
    paths = [
        os.path.join(output_dir, "metrics_valid.json"),
        os.path.join(output_dir, "backtest", "backtest_prediction_metrics.json"),
        os.path.join(output_dir, "backtest", "backtest_metrics.json"),
    ]

    for path in paths:
        if os.path.exists(path):
            print("=" * 80)
            print(path)
            print("=" * 80)
            metrics = load_json(path)
            for k, v in metrics.items():
                print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    output_dir = args.output_dir
    fig_dir = os.path.join(output_dir, "figures", "visualization")
    ensure_dir(fig_dir)

    print_metrics(output_dir)

    plot_training_loss(output_dir, fig_dir)

    valid_pred_path = os.path.join(output_dir, "predictions", "valid_predictions.csv")
    if os.path.exists(valid_pred_path):
        valid_pred = pd.read_csv(valid_pred_path)
        plot_daily_ic(valid_pred, fig_dir, prefix="Valid")
        plot_prediction_distribution(valid_pred, fig_dir, prefix="Valid")
        plot_decile_return(valid_pred, fig_dir, prefix="Valid")
    else:
        print(f"找不到验证集预测文件: {valid_pred_path}")

    backtest_pred_path = os.path.join(output_dir, "backtest", "backtest_predictions.csv")
    if os.path.exists(backtest_pred_path):
        backtest_pred = pd.read_csv(backtest_pred_path)
        plot_daily_ic(backtest_pred, fig_dir, prefix="Backtest")
        plot_prediction_distribution(backtest_pred, fig_dir, prefix="Backtest")
        plot_decile_return(backtest_pred, fig_dir, prefix="Backtest")
    else:
        print(f"找不到回测预测文件: {backtest_pred_path}")

    plot_nav_and_drawdown(output_dir, fig_dir)
    plot_turnover(output_dir, fig_dir)

    print("=" * 80)
    print(f"全部可视化完成，图片保存在: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()