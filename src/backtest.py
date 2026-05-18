import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from data_utils import load_config, prepare_data, save_json
from metrics import calc_backtest_metrics, calc_metrics
from model import MLPRegressor


def predict_df(model, df, feature_cols, batch_size, device):
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.extend(pred)
    return np.asarray(preds)


def topk_drop_backtest(pred_df, top_k=10, drop_k=2, initial_nav=1.0, fee_rate=0.0003):
    """
    简化回测：
    - 每天基于 pred 排名选股
    - 第一天买入 top_k
    - 之后每天卖出当前持仓中预测分数最低的 drop_k 只
    - 买入未持仓股票中预测分数最高的若干只，补足 top_k
    - 当日组合收益 = 持仓股票 label 的平均值
    - fee_rate 用换仓比例近似扣费
    """
    pred_df = pred_df.sort_values(["trade_date", "pred"], ascending=[True, False]).copy()

    dates = sorted(pred_df["trade_date"].unique())

    positions = []
    nav = initial_nav

    nav_records = []
    trade_records = []
    position_records = []

    for date in dates:
        day = pred_df[pred_df["trade_date"] == date].copy()
        day = day.sort_values("pred", ascending=False)

        score_map = dict(zip(day["ts_code"], day["pred"]))
        label_map = dict(zip(day["ts_code"], day["label"]))

        if len(positions) == 0:
            buy_list = day.head(top_k)["ts_code"].tolist()
            sell_list = []
            positions = buy_list
        else:
            holding_scores = [(c, score_map.get(c, -np.inf)) for c in positions]
            holding_scores = sorted(holding_scores, key=lambda x: x[1])

            sell_list = [c for c, _ in holding_scores[:drop_k]]

            remain = [c for c in positions if c not in sell_list]

            buy_list = []
            for c in day["ts_code"].tolist():
                if c not in remain and c not in buy_list:
                    buy_list.append(c)
                if len(remain) + len(buy_list) >= top_k:
                    break

            positions = remain + buy_list

        # 计算组合收益
        valid_rets = [label_map[c] for c in positions if c in label_map and pd.notna(label_map[c])]
        if len(valid_rets) == 0:
            port_ret = 0.0
        else:
            port_ret = float(np.mean(valid_rets))

        # 简化交易成本：买卖数量 / 持仓数量 * fee_rate * 2
        turnover = (len(sell_list) + len(buy_list)) / max(top_k, 1)
        cost = turnover * fee_rate
        net_ret = port_ret - cost

        nav = nav * (1 + net_ret)

        nav_records.append({
            "trade_date": date,
            "nav": nav,
            "portfolio_ret": port_ret,
            "cost": cost,
            "net_ret": net_ret,
            "num_positions": len(positions),
            "turnover": turnover,
        })

        for c in sell_list:
            trade_records.append({
                "trade_date": date,
                "action": "sell",
                "ts_code": c,
                "pred": score_map.get(c, np.nan),
            })

        for c in buy_list:
            trade_records.append({
                "trade_date": date,
                "action": "buy",
                "ts_code": c,
                "pred": score_map.get(c, np.nan),
            })

        weight = 1.0 / max(len(positions), 1)
        for c in positions:
            position_records.append({
                "trade_date": date,
                "ts_code": c,
                "weight": weight,
                "pred": score_map.get(c, np.nan),
                "label": label_map.get(c, np.nan),
            })

    nav_df = pd.DataFrame(nav_records)
    trade_df = pd.DataFrame(trade_records)
    pos_df = pd.DataFrame(position_records)

    return nav_df, trade_df, pos_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = cfg["output_dir"]
    os.makedirs(os.path.join(output_dir, "backtest"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    _, _, backtest_df, _ = prepare_data(cfg)
    if backtest_df.empty:
        raise ValueError("backtest_df 为空，请检查 backtest 时间范围。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(output_dir, "models", "best_mlp.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MLPRegressor(
        input_dim=ckpt["input_dim"],
        hidden_dims=cfg["train"]["hidden_dims"],
        dropout=cfg["train"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    preds = predict_df(
        model,
        backtest_df,
        feature_cols,
        cfg["train"]["batch_size"],
        device,
    )

    pred_df = backtest_df[["ts_code", "trade_date", label_col]].copy()
    pred_df = pred_df.rename(columns={label_col: "label"})
    pred_df["pred"] = preds

    pred_path = os.path.join(output_dir, "backtest", "backtest_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    pred_metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    save_json(pred_metrics, os.path.join(output_dir, "backtest", "backtest_prediction_metrics.json"))

    bt_cfg = cfg["backtest"]
    nav_df, trade_df, pos_df = topk_drop_backtest(
        pred_df,
        top_k=bt_cfg["top_k"],
        drop_k=bt_cfg["drop_k"],
        initial_nav=bt_cfg["initial_nav"],
        fee_rate=bt_cfg["fee_rate"],
    )

    nav_df.to_csv(os.path.join(output_dir, "backtest", "nav_curve.csv"), index=False, encoding="utf-8-sig")
    trade_df.to_csv(os.path.join(output_dir, "backtest", "trade_records.csv"), index=False, encoding="utf-8-sig")
    pos_df.to_csv(os.path.join(output_dir, "backtest", "position_records.csv"), index=False, encoding="utf-8-sig")

    metrics = calc_backtest_metrics(nav_df)
    save_json(metrics, os.path.join(output_dir, "backtest", "backtest_metrics.json"))

    # 净值图
    plt.figure(figsize=(9, 5))
    plt.plot(nav_df["trade_date"].astype(str), nav_df["nav"])
    plt.xticks(nav_df["trade_date"].astype(str)[::max(len(nav_df)//10, 1)], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("MLP TopK-Drop Backtest NAV")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "backtest_nav.png"), dpi=200)
    plt.close()

    # 回撤图
    nav = nav_df["nav"]
    drawdown = nav / nav.cummax() - 1

    plt.figure(figsize=(9, 5))
    plt.plot(nav_df["trade_date"].astype(str), drawdown)
    plt.xticks(nav_df["trade_date"].astype(str)[::max(len(nav_df)//10, 1)], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("MLP TopK-Drop Backtest Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "backtest_drawdown.png"), dpi=200)
    plt.close()

    print("=" * 80)
    print("Backtest prediction metrics:")
    for k, v in pred_metrics.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("Backtest performance metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("回测完成")
    print(f"预测文件: {pred_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
