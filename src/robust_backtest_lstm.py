import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from data_utils import load_config, prepare_data, load_feature_data, save_json
from metrics import calc_metrics
from model_lstm import LSTMRegressor
from sequence_dataset import StockSequenceDataset


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def predict_sequence(model, ds, batch_size, device, num_workers=0):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    preds, labels, codes, dates = [], [], [], []

    model.eval()
    with torch.no_grad():
        for x, y, ts_code, trade_date in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
            labels.append(y.numpy())
            codes.extend(list(ts_code))
            dates.extend([int(d) for d in trade_date])

    return pd.DataFrame({
        "ts_code": codes,
        "signal_date": dates,
        "model_label": np.concatenate(labels),
        "pred": np.concatenate(preds),
    })


def make_signal_frame(cfg, model_path):
    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]
    seq_len = cfg["sequence"]["seq_len"]

    _, _, bt_std, _ = prepare_data(cfg)

    raw = load_feature_data(cfg["data_path"])
    s = cfg["split"]
    raw = raw[(raw["trade_date"] >= s["backtest_start"]) & (raw["trade_date"] <= s["backtest_end"])].copy()

    keep_cols = [
        "ts_code", "trade_date", "open", "close",
        "ret_1", "ret_5", "vol_std_20", "amount_ma20"
    ]
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    bt_ds = StockSequenceDataset(bt_std, feature_cols, label_col, seq_len=seq_len, return_meta=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    model = LSTMRegressor(
        input_dim=ckpt["input_dim"],
        hidden_size=cfg["train"]["hidden_size"],
        num_layers=cfg["train"]["num_layers"],
        dropout=cfg["train"]["dropout"],
        bidirectional=cfg["train"].get("bidirectional", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    sig = predict_sequence(
        model,
        bt_ds,
        batch_size=cfg["train"]["batch_size"],
        device=device,
        num_workers=cfg["train"].get("num_workers", 0),
    )

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

        day["score"] = rng.normal(size=len(day)) if strategy.startswith("random") else day["pred"]
        day = day.sort_values("score", ascending=False)

        ret_map = dict(zip(day["ts_code"], day["realized_ret"]))
        score_map = dict(zip(day["ts_code"], day["score"]))

        sell_list, buy_list = [], []

        if strategy in ["lstm_top10_full", "random_top10_full"]:
            target = day.head(top_k)["ts_code"].tolist()
            sell_list = [c for c in positions if c not in target]
            buy_list = [c for c in target if c not in positions]
            positions = target

        elif strategy in ["lstm_top10_drop2", "random_top10_drop2"]:
            if len(positions) == 0:
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

        elif strategy == "lstm_buffer_risk":
            ranked = day.copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            rank_map = dict(zip(ranked["ts_code"], ranked["rank"]))

            buy_pool = ranked[ranked["rank"] <= 20].copy()
            buy_pool = apply_buy_filter(buy_pool)
            buy_pool = buy_pool.sort_values("score", ascending=False)

            if len(positions) == 0:
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
            raise ValueError(f"未知策略: {strategy}")

        valid_rets = [ret_map[c] for c in positions if c in ret_map and pd.notna(ret_map[c])]
        port_ret = float(np.mean(valid_rets)) if len(valid_rets) else 0.0
        turnover = (len(sell_list) + len(buy_list)) / max(top_k, 1)
        cost = turnover * fee_rate
        net_ret = port_ret - cost
        nav *= (1 + net_ret)

        nav_records.append({
            "trade_date": int(d),
            "nav": nav,
            "portfolio_ret": port_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": turnover,
            "num_positions": len(positions),
        })

        for c in sell_list:
            trade_records.append({"trade_date": int(d), "strategy": strategy, "action": "sell", "ts_code": c, "score": score_map.get(c, np.nan)})
        for c in buy_list:
            trade_records.append({"trade_date": int(d), "strategy": strategy, "action": "buy", "ts_code": c, "score": score_map.get(c, np.nan)})

        w = 1 / max(len(positions), 1)
        for c in positions:
            pos_records.append({
                "trade_date": int(d), "strategy": strategy, "ts_code": c,
                "weight": w, "score": score_map.get(c, np.nan),
                "realized_ret": ret_map.get(c, np.nan)
            })

    return pd.DataFrame(nav_records), pd.DataFrame(trade_records), pd.DataFrame(pos_records)


def perf_metrics(nav_df):
    nav = nav_df["nav"].astype(float)
    ret = nav_df["net_ret"].astype(float)

    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "annual_return": float((nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav_df)) - 1),
        "annual_vol": float(ret.std() * np.sqrt(252)),
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() != 0 else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "win_rate": float((ret > 0).mean()),
        "avg_turnover": float(nav_df["turnover"].mean()),
        "num_days": int(len(nav_df)),
    }


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
    plt.title("LSTM Strategy NAV Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_nav_comparison_lstm.png"), dpi=200)
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
    plt.title("LSTM Strategy Drawdown Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_drawdown_comparison_lstm.png"), dpi=200)
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

    monthly.to_csv(os.path.join(out_dir, "monthly_ic_lstm.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.bar(monthly["month"].astype(str), monthly["monthly_ic"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Monthly IC")
    plt.title("LSTM Monthly IC")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monthly_ic_lstm.png"), dpi=200)
    plt.close()

    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_lstm.json")
    parser.add_argument("--model_path", type=str, default="outputs_lstm/models/best_lstm.pt")
    parser.add_argument("--out_dir", type=str, default="outputs_lstm/robust_backtest_lstm")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--drop_k", type=int, default=2)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--random_runs", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cfg = load_config(args.config)

    print("=" * 80)
    print("生成 LSTM realistic signal predictions")
    print("=" * 80)

    sig = make_signal_frame(cfg, args.model_path)
    sig.to_csv(os.path.join(args.out_dir, "realistic_signal_predictions_lstm.csv"), index=False, encoding="utf-8-sig")

    pred_df = sig[["ts_code", "exec_date", "pred", "realized_ret"]].rename(
        columns={"exec_date": "trade_date", "realized_ret": "label"}
    )

    pred_metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    save_json(pred_metrics, os.path.join(args.out_dir, "realistic_prediction_metrics_lstm.json"))
    monthly_ic_plot(pred_df, args.out_dir)

    print("=" * 80)
    print("LSTM Realistic prediction metrics")
    print("=" * 80)
    for k, v in pred_metrics.items():
        print(f"{k}: {v}")

    nav_dict = {}
    metrics = []

    for st in ["lstm_top10_full", "lstm_top10_drop2", "lstm_buffer_risk"]:
        nav_df, trade_df, pos_df = run_strategy(sig, st, args.top_k, args.drop_k, args.fee_rate, seed=42)
        nav_dict[st] = nav_df

        nav_df.to_csv(os.path.join(args.out_dir, f"{st}_nav.csv"), index=False, encoding="utf-8-sig")
        trade_df.to_csv(os.path.join(args.out_dir, f"{st}_trades.csv"), index=False, encoding="utf-8-sig")
        pos_df.to_csv(os.path.join(args.out_dir, f"{st}_positions.csv"), index=False, encoding="utf-8-sig")

        m = perf_metrics(nav_df)
        m["strategy"] = st
        m["run"] = 0
        metrics.append(m)

    for st in ["random_top10_full", "random_top10_drop2"]:
        sample_nav = None
        for r in range(args.random_runs):
            nav_df, _, _ = run_strategy(sig, st, args.top_k, args.drop_k, args.fee_rate, seed=1000 + r)
            if r == 0:
                sample_nav = nav_df
                nav_df.to_csv(os.path.join(args.out_dir, f"{st}_sample_nav.csv"), index=False, encoding="utf-8-sig")

            m = perf_metrics(nav_df)
            m["strategy"] = st
            m["run"] = r
            metrics.append(m)

        nav_dict[st + "_sample"] = sample_nav

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.out_dir, "strategy_metrics_all_runs_lstm.csv"), index=False, encoding="utf-8-sig")

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
    summary.to_csv(os.path.join(args.out_dir, "strategy_metrics_summary_lstm.csv"), index=False, encoding="utf-8-sig")

    plot_nav(nav_dict, args.out_dir)

    print("=" * 80)
    print("LSTM Strategy summary")
    print("=" * 80)
    print(summary)
    print("=" * 80)
    print("完成，输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
