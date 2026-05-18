
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR / "src"))
sys.path.append(str(THIS_DIR.parent / "src"))

from data_utils import load_config, prepare_data, load_feature_data, save_json
from model import MLPRegressor
from metrics import calc_metrics


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def predict_df(model, df, feature_cols, batch_size, device):
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds)


def perf_metrics(nav_df):
    nav = nav_df["nav"].astype(float)
    ret = nav_df["net_ret"].astype(float)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav_df)) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
    max_drawdown = (nav / nav.cummax() - 1).min()
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float((ret > 0).mean()),
        "avg_turnover": float(nav_df["turnover"].mean()),
        "num_days": int(len(nav_df)),
    }


def make_signal_frame(cfg, model_path):
    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    # 标准化后的 backtest 数据，用于模型预测
    _, _, bt_std, _ = prepare_data(cfg)

    # 未标准化原始数据，用于 open/close 和风控字段
    raw = load_feature_data(cfg["data_path"])
    s = cfg["split"]
    raw = raw[(raw["trade_date"] >= s["backtest_start"]) & (raw["trade_date"] <= s["backtest_end"])].copy()

    keep_cols = [
        "ts_code", "trade_date", "open", "close", "ret_1", "ret_5",
        "vol_std_20", "amount_ma20", "amplitude", "body_ratio"
    ]
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = MLPRegressor(
        input_dim=ckpt["input_dim"],
        hidden_dims=cfg["train"]["hidden_dims"],
        dropout=cfg["train"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred = predict_df(model, bt_std, feature_cols, cfg["train"]["batch_size"], device)

    sig = bt_std[["ts_code", "trade_date", label_col]].copy()
    sig = sig.rename(columns={"trade_date": "signal_date", label_col: "close_to_close_label"})
    sig["pred"] = pred

    raw_signal = raw.rename(columns={"trade_date": "signal_date"})
    sig = sig.merge(raw_signal, on=["ts_code", "signal_date"], how="left")

    dates = sorted(raw["trade_date"].unique())
    date_map = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    sig["exec_date"] = sig["signal_date"].map(date_map)
    sig = sig.dropna(subset=["exec_date"]).copy()
    sig["exec_date"] = sig["exec_date"].astype(int)

    exec_price = raw[["ts_code", "trade_date", "open", "close"]].copy()
    exec_price = exec_price.rename(columns={
        "trade_date": "exec_date",
        "open": "exec_open",
        "close": "exec_close",
    })
    sig = sig.merge(exec_price, on=["ts_code", "exec_date"], how="inner")

    sig["realized_ret"] = sig["exec_close"] / sig["exec_open"] - 1
    sig = sig.replace([np.inf, -np.inf], np.nan)
    sig = sig.dropna(subset=["pred", "realized_ret"])
    return sig


def apply_buy_filter(day):
    x = day.copy()
    if "ret_1" in x.columns:
        x = x[x["ret_1"].abs() < 0.08]
    if "ret_5" in x.columns:
        x = x[(x["ret_5"] < 0.20) & (x["ret_5"] > -0.20)]
    if "vol_std_20" in x.columns and x["vol_std_20"].notna().sum() > 10:
        x = x[x["vol_std_20"] < x["vol_std_20"].quantile(0.90)]
    if "amount_ma20" in x.columns and x["amount_ma20"].notna().sum() > 10:
        x = x[x["amount_ma20"] > x["amount_ma20"].quantile(0.20)]
    return x


def run_strategy(sig, strategy, top_k=10, drop_k=2, fee_rate=0.0003, seed=42):
    rng = np.random.default_rng(seed)
    dates = sorted(sig["exec_date"].unique())
    positions = []
    nav = 1.0
    nav_records, trade_records, pos_records = [], [], []

    for d in dates:
        day = sig[sig["exec_date"] == d].copy()
        if day.empty:
            continue

        if strategy.startswith("random"):
            day["score"] = rng.normal(size=len(day))
        else:
            day["score"] = day["pred"]

        day = day.sort_values("score", ascending=False)
        ret_map = dict(zip(day["ts_code"], day["realized_ret"]))
        score_map = dict(zip(day["ts_code"], day["score"]))

        sell_list, buy_list = [], []

        if strategy in ["mlp_top10_full", "random_top10_full"]:
            target = day.head(top_k)["ts_code"].tolist()
            sell_list = [c for c in positions if c not in target]
            buy_list = [c for c in target if c not in positions]
            positions = target

        elif strategy in ["mlp_top10_drop2", "random_top10_drop2"]:
            if not positions:
                positions = day.head(top_k)["ts_code"].tolist()
                buy_list = positions.copy()
            else:
                holding_scores = sorted([(c, score_map.get(c, -np.inf)) for c in positions], key=lambda x: x[1])
                sell_list = [c for c, _ in holding_scores[:drop_k]]
                remain = [c for c in positions if c not in sell_list]
                buy_list = []
                for c in day["ts_code"].tolist():
                    if c not in remain and c not in buy_list:
                        buy_list.append(c)
                    if len(remain) + len(buy_list) >= top_k:
                        break
                positions = remain + buy_list

        elif strategy == "mlp_buffer_risk":
            ranked = day.copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            rank_map = dict(zip(ranked["ts_code"], ranked["rank"]))

            buy_pool = ranked[ranked["rank"] <= 20].copy()
            buy_pool = apply_buy_filter(buy_pool)
            buy_pool = buy_pool.sort_values("score", ascending=False)

            if not positions:
                positions = buy_pool.head(top_k)["ts_code"].tolist()
                buy_list = positions.copy()
            else:
                sell_candidates = []
                for c in positions:
                    r = rank_map.get(c, 999999)
                    if r > 50:
                        sell_candidates.append((c, r))
                sell_candidates = sorted(sell_candidates, key=lambda x: x[1], reverse=True)
                sell_list = [c for c, _ in sell_candidates[:3]]

                remain = [c for c in positions if c not in sell_list]
                buy_list = []
                for c in buy_pool["ts_code"].tolist():
                    if c not in remain and c not in buy_list:
                        buy_list.append(c)
                    if len(remain) + len(buy_list) >= top_k:
                        break
                positions = remain + buy_list
        else:
            raise ValueError(strategy)

        valid_rets = [ret_map[c] for c in positions if c in ret_map and pd.notna(ret_map[c])]
        port_ret = float(np.mean(valid_rets)) if valid_rets else 0.0
        turnover = (len(sell_list) + len(buy_list)) / max(top_k, 1)
        cost = turnover * fee_rate
        net_ret = port_ret - cost
        nav *= (1 + net_ret)

        nav_records.append({
            "trade_date": int(d), "nav": nav, "portfolio_ret": port_ret,
            "cost": cost, "net_ret": net_ret, "turnover": turnover,
            "num_positions": len(positions),
        })

        for c in sell_list:
            trade_records.append({"trade_date": int(d), "strategy": strategy, "action": "sell", "ts_code": c, "score": score_map.get(c, np.nan)})
        for c in buy_list:
            trade_records.append({"trade_date": int(d), "strategy": strategy, "action": "buy", "ts_code": c, "score": score_map.get(c, np.nan)})

        w = 1 / max(len(positions), 1)
        for c in positions:
            pos_records.append({"trade_date": int(d), "strategy": strategy, "ts_code": c, "weight": w, "score": score_map.get(c, np.nan), "realized_ret": ret_map.get(c, np.nan)})

    return pd.DataFrame(nav_records), pd.DataFrame(trade_records), pd.DataFrame(pos_records)


def plot_nav(nav_dict, out_dir):
    plt.figure(figsize=(11, 6))
    for name, df in nav_dict.items():
        if not df.empty:
            plt.plot(df["trade_date"].astype(str), df["nav"], label=name)
    first = next(iter(nav_dict.values()))
    if not first.empty:
        x = first["trade_date"].astype(str)
        plt.xticks(x[::max(len(x)//10, 1)], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Realistic Strategy NAV Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_nav_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, df in nav_dict.items():
        if not df.empty:
            nav = df["nav"].astype(float)
            dd = nav / nav.cummax() - 1
            plt.plot(df["trade_date"].astype(str), dd, label=name)
    if not first.empty:
        x = first["trade_date"].astype(str)
        plt.xticks(x[::max(len(x)//10, 1)], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Realistic Strategy Drawdown Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_drawdown_comparison.png"), dpi=200)
    plt.close()


def monthly_ic_plot(pred_df, out_dir):
    pred_df = pred_df.copy()
    pred_df["month"] = pred_df["trade_date"].astype(str).str[:6]
    monthly_ic = pred_df.groupby("month").apply(lambda x: x["pred"].corr(x["label"])).dropna()
    monthly_rank_ic = pred_df.groupby("month").apply(lambda x: x["pred"].corr(x["label"], method="spearman")).dropna()

    monthly = pd.DataFrame({
        "month": monthly_ic.index,
        "monthly_ic": monthly_ic.values,
        "monthly_rank_ic": monthly_rank_ic.reindex(monthly_ic.index).values,
    })
    monthly.to_csv(os.path.join(out_dir, "monthly_ic.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.bar(monthly["month"].astype(str), monthly["monthly_ic"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Monthly IC")
    plt.title("Monthly IC, next-day open-to-close return")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monthly_ic.png"), dpi=200)
    plt.close()

    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--model_path", type=str, default="outputs/models/best_mlp.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/robust_backtest")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--drop_k", type=int, default=2)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--random_runs", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cfg = load_config(args.config)

    print("=" * 80)
    print("生成更真实的 next-day open-to-close 预测表")
    print("=" * 80)
    sig = make_signal_frame(cfg, args.model_path)
    sig.to_csv(os.path.join(args.out_dir, "realistic_signal_predictions.csv"), index=False, encoding="utf-8-sig")
    print("signal shape:", sig.shape)

    pred_df = sig[["ts_code", "exec_date", "pred", "realized_ret"]].rename(columns={"exec_date": "trade_date", "realized_ret": "label"})
    pred_metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    save_json(pred_metrics, os.path.join(args.out_dir, "realistic_prediction_metrics.json"))
    monthly = monthly_ic_plot(pred_df, args.out_dir)

    print("=" * 80)
    print("Realistic prediction metrics")
    print("=" * 80)
    for k, v in pred_metrics.items():
        print(f"{k}: {v}")

    strategies = ["mlp_top10_full", "mlp_top10_drop2", "mlp_buffer_risk"]
    nav_dict = {}
    metrics = []

    for st in strategies:
        nav_df, trade_df, pos_df = run_strategy(sig, st, args.top_k, args.drop_k, args.fee_rate, seed=42)
        nav_dict[st] = nav_df
        nav_df.to_csv(os.path.join(args.out_dir, f"{st}_nav.csv"), index=False, encoding="utf-8-sig")
        trade_df.to_csv(os.path.join(args.out_dir, f"{st}_trades.csv"), index=False, encoding="utf-8-sig")
        pos_df.to_csv(os.path.join(args.out_dir, f"{st}_positions.csv"), index=False, encoding="utf-8-sig")
        m = perf_metrics(nav_df)
        m["strategy"] = st
        m["run"] = 0
        metrics.append(m)

    # 随机策略多次
    for st in ["random_top10_full", "random_top10_drop2"]:
        sample_nav = None
        for r in range(args.random_runs):
            nav_df, _, _ = run_strategy(sig, st, args.top_k, args.drop_k, args.fee_rate, seed=1000+r)
            if r == 0:
                sample_nav = nav_df
                nav_df.to_csv(os.path.join(args.out_dir, f"{st}_sample_nav.csv"), index=False, encoding="utf-8-sig")
            m = perf_metrics(nav_df)
            m["strategy"] = st
            m["run"] = r
            metrics.append(m)
        nav_dict[st + "_sample"] = sample_nav

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.out_dir, "strategy_metrics_all_runs.csv"), index=False, encoding="utf-8-sig")

    summary = metrics_df.groupby("strategy").agg({
        "total_return": ["mean", "std"],
        "annual_return": ["mean", "std"],
        "sharpe": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "win_rate": ["mean", "std"],
        "avg_turnover": ["mean", "std"],
        "num_days": "mean",
    })
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out_dir, "strategy_metrics_summary.csv"), index=False, encoding="utf-8-sig")

    plot_nav(nav_dict, args.out_dir)

    print("=" * 80)
    print("Strategy summary")
    print("=" * 80)
    print(summary)
    print("=" * 80)
    print("完成，输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
