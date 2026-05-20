import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from data_utils import load_config, prepare_data, load_feature_data, save_json
from metrics import calc_metrics


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_model_name(model_type, model_path):
    p = str(model_path).lower()
    if model_type == "mlp":
        return "mlp_oo"
    if model_type == "lstm":
        if "seq20" in p:
            return "lstm_seq20_oo"
        if "seq10" in p:
            return "lstm_seq10_oo"
        return "lstm_oo"
    if model_type == "dlinear":
        if "seq20" in p:
            return "dlinear_seq20_oo"
        if "seq10" in p:
            return "dlinear_seq10_oo"
        return "dlinear_oo"
    return model_type


def performance_metrics(nav_df):
    if len(nav_df) == 0:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "avg_turnover": np.nan,
            "num_days": 0,
        }

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


def load_model(model_type, cfg, model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    if model_type == "mlp":
        from model import MLPRegressor
        model = MLPRegressor(
            input_dim=ckpt["input_dim"],
            hidden_dims=cfg["train"]["hidden_dims"],
            dropout=cfg["train"]["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    if model_type == "lstm":
        from model_lstm import LSTMRegressor
        model = LSTMRegressor(
            input_dim=ckpt["input_dim"],
            hidden_size=cfg["train"]["hidden_size"],
            num_layers=cfg["train"]["num_layers"],
            dropout=cfg["train"]["dropout"],
            bidirectional=cfg["train"].get("bidirectional", False),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    if model_type == "dlinear":
        from model_dlinear import DLinearRegressor
        model = DLinearRegressor(
            seq_len=ckpt["seq_len"],
            feature_dim=ckpt["feature_dim"],
            moving_avg=ckpt.get("moving_avg", cfg.get("sequence", {}).get("moving_avg", 3)),
            dropout=cfg["train"].get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_mlp(model, df, feature_cols, batch_size, device):
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds)


def predict_sequence_model(model, df, cfg, device):
    from sequence_dataset import StockSequenceDataset

    ds = StockSequenceDataset(
        df,
        cfg["feature_cols"],
        cfg["label_col"],
        seq_len=cfg["sequence"]["seq_len"],
        return_meta=True,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
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


def make_t1_signal_frame(cfg, model_type, model_path):
    """
    严格 T+1 open-to-open 回测信号表。

    signal_date = t
    buy_date    = t+1 open
    sell_date   = t+2 open
    realized_ret = open[t+2] / open[t+1] - 1

    要求 cfg["label_col"] = "label_oo_1d"。
    """

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    if label_col != "label_oo_1d":
        print(f"WARNING: 当前 label_col={label_col}，建议 T+1 实验使用 label_oo_1d")

    print("=" * 80)
    print("prepare standardized data")
    print("=" * 80)

    _, _, bt_std, _ = prepare_data(cfg)

    print("=" * 80)
    print("load raw open prices")
    print("=" * 80)

    raw = load_feature_data(cfg["data_path"])
    s = cfg["split"]
    raw = raw[
        (raw["trade_date"] >= s["backtest_start"]) &
        (raw["trade_date"] <= s["backtest_end"])
    ].copy()

    keep_cols = [
        "ts_code", "trade_date", "open", "close",
        "ret_1", "ret_5", "vol_std_20", "amount_ma20"
    ]
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(model_type, cfg, model_path, device)

    print("=" * 80)
    print("predict")
    print("=" * 80)

    if model_type == "mlp":
        preds = predict_mlp(
            model,
            bt_std,
            feature_cols,
            cfg["train"]["batch_size"],
            device,
        )

        sig = bt_std[["ts_code", "trade_date", label_col]].copy()
        sig = sig.rename(columns={
            "trade_date": "signal_date",
            label_col: "model_label",
        })
        sig["pred"] = preds

    else:
        sig = predict_sequence_model(model, bt_std, cfg, device)

    raw_signal = raw.rename(columns={"trade_date": "signal_date"})
    sig = sig.merge(raw_signal, on=["ts_code", "signal_date"], how="left")

    dates = sorted(raw["trade_date"].unique())

    buy_map = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    sell_map = {dates[i]: dates[i + 2] for i in range(len(dates) - 2)}

    sig["buy_date"] = sig["signal_date"].map(buy_map)
    sig["sell_date"] = sig["signal_date"].map(sell_map)

    sig = sig.dropna(subset=["buy_date", "sell_date"]).copy()
    sig["buy_date"] = sig["buy_date"].astype(int)
    sig["sell_date"] = sig["sell_date"].astype(int)

    buy_price = raw[["ts_code", "trade_date", "open"]].rename(
        columns={"trade_date": "buy_date", "open": "buy_open"}
    )
    sell_price = raw[["ts_code", "trade_date", "open"]].rename(
        columns={"trade_date": "sell_date", "open": "sell_open"}
    )

    sig = sig.merge(buy_price, on=["ts_code", "buy_date"], how="inner")
    sig = sig.merge(sell_price, on=["ts_code", "sell_date"], how="inner")

    sig["realized_ret"] = sig["sell_open"] / sig["buy_open"] - 1

    sig = sig.replace([np.inf, -np.inf], np.nan)
    sig = sig.dropna(subset=["pred", "realized_ret", "buy_open", "sell_open"])

    print("=" * 80)
    print("signal frame")
    print("=" * 80)
    print("shape:", sig.shape)
    print("buy_date:", sig["buy_date"].min(), "-", sig["buy_date"].max())
    print("num_days:", sig["buy_date"].nunique())
    print("num_stocks:", sig["ts_code"].nunique())

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


def run_strategy(sig, strategy_name, strategy_kind, top_k=10, drop_k=2, fee_rate=0.0003, seed=42):
    rng = np.random.default_rng(seed)
    dates = sorted(sig["buy_date"].unique())

    positions = []
    nav = 1.0

    nav_records = []
    trade_records = []
    position_records = []

    for d in dates:
        day = sig[sig["buy_date"] == d].copy()
        if day.empty:
            continue

        if strategy_name.startswith("random"):
            day["score"] = rng.normal(size=len(day))
        else:
            day["score"] = day["pred"]

        day = day.sort_values("score", ascending=False)

        ret_map = dict(zip(day["ts_code"], day["realized_ret"]))
        score_map = dict(zip(day["ts_code"], day["score"]))

        sell_list = []
        buy_list = []

        if strategy_kind == "top10_full":
            target = day.head(top_k)["ts_code"].tolist()
            sell_list = [c for c in positions if c not in target]
            buy_list = [c for c in target if c not in positions]
            positions = target

        elif strategy_kind == "top10_drop2":
            if len(positions) == 0:
                positions = day.head(top_k)["ts_code"].tolist()
                buy_list = positions.copy()
            else:
                holding_scores = sorted(
                    [(c, score_map.get(c, -np.inf)) for c in positions],
                    key=lambda x: x[1],
                )
                sell_list = [c for c, _ in holding_scores[:drop_k]]
                remain = [c for c in positions if c not in sell_list]

                buy_list = []
                for c in day["ts_code"].tolist():
                    if c not in remain and c not in buy_list:
                        buy_list.append(c)
                    if len(remain) + len(buy_list) >= top_k:
                        break

                positions = remain + buy_list

        elif strategy_kind == "buffer_risk":
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
            raise ValueError(f"Unknown strategy_kind: {strategy_kind}")

        valid_rets = [
            ret_map[c]
            for c in positions
            if c in ret_map and pd.notna(ret_map[c])
        ]

        portfolio_ret = float(np.mean(valid_rets)) if len(valid_rets) > 0 else 0.0

        turnover = (len(sell_list) + len(buy_list)) / max(top_k, 1)
        cost = turnover * fee_rate
        net_ret = portfolio_ret - cost
        nav *= (1 + net_ret)

        nav_records.append({
            "trade_date": int(d),
            "nav": nav,
            "portfolio_ret": portfolio_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": turnover,
            "num_positions": len(positions),
        })

        for c in sell_list:
            trade_records.append({
                "trade_date": int(d),
                "strategy": strategy_name,
                "action": "sell",
                "ts_code": c,
                "score": score_map.get(c, np.nan),
            })

        for c in buy_list:
            trade_records.append({
                "trade_date": int(d),
                "strategy": strategy_name,
                "action": "buy",
                "ts_code": c,
                "score": score_map.get(c, np.nan),
            })

        weight = 1 / max(len(positions), 1)

        for c in positions:
            position_records.append({
                "trade_date": int(d),
                "strategy": strategy_name,
                "ts_code": c,
                "weight": weight,
                "score": score_map.get(c, np.nan),
                "realized_ret": ret_map.get(c, np.nan),
            })

    return pd.DataFrame(nav_records), pd.DataFrame(trade_records), pd.DataFrame(position_records)


def plot_nav_and_drawdown(nav_dict, out_dir):
    plt.figure(figsize=(11, 6))

    for name, df in nav_dict.items():
        if df is not None and not df.empty:
            plt.plot(df["trade_date"].astype(str), df["nav"], label=name)

    first = next(iter(nav_dict.values()))
    if first is not None and not first.empty:
        x = first["trade_date"].astype(str)
        plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("T+1 Open-to-Open NAV Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_nav_comparison_t1.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))

    for name, df in nav_dict.items():
        if df is not None and not df.empty:
            nav = df["nav"].astype(float)
            dd = nav / nav.cummax() - 1
            plt.plot(df["trade_date"].astype(str), dd, label=name)

    if first is not None and not first.empty:
        x = first["trade_date"].astype(str)
        plt.xticks(x[::max(len(x) // 10, 1)], rotation=45)

    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("T+1 Open-to-Open Drawdown Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strategy_drawdown_comparison_t1.png"), dpi=200)
    plt.close()


def monthly_ic_plot(pred_df, out_dir):
    pred_df = pred_df.copy()
    pred_df["month"] = pred_df["trade_date"].astype(str).str[:6]

    monthly_ic = pred_df.groupby("month").apply(
        lambda x: x["pred"].corr(x["label"])
    ).dropna()

    monthly_rank_ic = pred_df.groupby("month").apply(
        lambda x: x["pred"].corr(x["label"], method="spearman")
    ).dropna()

    monthly = pd.DataFrame({
        "month": monthly_ic.index,
        "monthly_ic": monthly_ic.values,
        "monthly_rank_ic": monthly_rank_ic.reindex(monthly_ic.index).values,
    })

    monthly.to_csv(
        os.path.join(out_dir, "monthly_ic_t1.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 5))
    plt.bar(monthly["month"].astype(str), monthly["monthly_ic"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("Monthly IC")
    plt.title("T+1 Open-to-Open Monthly IC")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monthly_ic_t1.png"), dpi=200)
    plt.close()

    return monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["mlp", "lstm", "dlinear"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--drop_k", type=int, default=2)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--random_runs", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    cfg = load_config(args.config)
    model_name = args.model_name or infer_model_name(args.model_type, args.model_path)

    print("=" * 80)
    print("T+1 open-to-open backtest")
    print("=" * 80)
    print("model_type:", args.model_type)
    print("model_name:", model_name)
    print("config:", args.config)
    print("model_path:", args.model_path)
    print("out_dir:", args.out_dir)

    sig = make_t1_signal_frame(cfg, args.model_type, args.model_path)

    sig_path = os.path.join(args.out_dir, "t1_signal_predictions.csv")
    sig.to_csv(sig_path, index=False, encoding="utf-8-sig")

    pred_df = sig[["ts_code", "buy_date", "pred", "realized_ret"]].rename(
        columns={
            "buy_date": "trade_date",
            "realized_ret": "label",
        }
    )

    pred_metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    save_json(pred_metrics, os.path.join(args.out_dir, "t1_prediction_metrics.json"))
    monthly_ic_plot(pred_df, args.out_dir)

    print("=" * 80)
    print("T+1 Open-to-Open prediction metrics")
    print("=" * 80)
    for k, v in pred_metrics.items():
        print(f"{k}: {v}")

    nav_dict = {}
    metrics = []

    for kind in ["top10_full", "top10_drop2", "buffer_risk"]:
        strategy_name = f"{model_name}_{kind}"

        nav_df, trade_df, pos_df = run_strategy(
            sig=sig,
            strategy_name=strategy_name,
            strategy_kind=kind,
            top_k=args.top_k,
            drop_k=args.drop_k,
            fee_rate=args.fee_rate,
            seed=42,
        )

        nav_dict[strategy_name] = nav_df

        nav_df.to_csv(
            os.path.join(args.out_dir, f"{strategy_name}_nav.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        trade_df.to_csv(
            os.path.join(args.out_dir, f"{strategy_name}_trades.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pos_df.to_csv(
            os.path.join(args.out_dir, f"{strategy_name}_positions.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        m = performance_metrics(nav_df)
        m["strategy"] = strategy_name
        m["run"] = 0
        metrics.append(m)

    for kind in ["top10_full", "top10_drop2"]:
        random_name = f"random_{kind}"
        sample_nav = None

        for r in range(args.random_runs):
            nav_df, _, _ = run_strategy(
                sig=sig,
                strategy_name=random_name,
                strategy_kind=kind,
                top_k=args.top_k,
                drop_k=args.drop_k,
                fee_rate=args.fee_rate,
                seed=1000 + r,
            )

            if r == 0:
                sample_nav = nav_df
                nav_df.to_csv(
                    os.path.join(args.out_dir, f"{random_name}_sample_nav.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

            m = performance_metrics(nav_df)
            m["strategy"] = random_name
            m["run"] = r
            metrics.append(m)

        nav_dict[random_name + "_sample"] = sample_nav

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        os.path.join(args.out_dir, "strategy_metrics_all_runs_t1.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = metrics_df.groupby("strategy").agg({
        "total_return": ["mean", "std"],
        "annual_return": ["mean", "std"],
        "annual_vol": ["mean", "std"],
        "sharpe": ["mean", "std"],
        "max_drawdown": ["mean", "std"],
        "win_rate": ["mean", "std"],
        "avg_turnover": ["mean", "std"],
        "num_days": "mean",
    })

    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    summary.to_csv(
        os.path.join(args.out_dir, "strategy_metrics_summary_t1.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_nav_and_drawdown(nav_dict, args.out_dir)

    print("=" * 80)
    print("T+1 Open-to-Open Strategy summary")
    print("=" * 80)
    print(summary)
    print("=" * 80)
    print("完成，输出目录:", args.out_dir)


if __name__ == "__main__":
    main()
