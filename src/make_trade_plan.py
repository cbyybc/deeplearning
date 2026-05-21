import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    required = {"ts_code", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")

    df["ts_code"] = df["ts_code"].astype(str)

    if "rank" not in df.columns:
        df = df.sort_values("pred", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df = df.sort_values("rank", ascending=True).reset_index(drop=True)

    if "signal_date" not in df.columns:
        df["signal_date"] = ""

    return df


def read_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "buy_date", "shares", "weight"])

    df = pd.read_csv(path)

    if len(df) == 0:
        return pd.DataFrame(columns=["ts_code", "buy_date", "shares", "weight"])

    if "ts_code" not in df.columns:
        raise ValueError("Position file must contain column: ts_code")

    df["ts_code"] = df["ts_code"].astype(str)

    if "buy_date" not in df.columns:
        df["buy_date"] = ""
    if "shares" not in df.columns:
        df["shares"] = np.nan
    if "weight" not in df.columns:
        df["weight"] = np.nan

    return df[["ts_code", "buy_date", "shares", "weight"]].copy()


def normalize_trade_date(trade_date: str | None) -> str:
    if trade_date is None or trade_date == "":
        return datetime.now().strftime("%Y-%m-%d")
    return pd.to_datetime(trade_date).strftime("%Y-%m-%d")


def attach_prediction_to_positions(positions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ["ts_code", "pred", "rank", "signal_date"]
    merged = positions.merge(candidates[pred_cols], on="ts_code", how="left")

    # 如果当前持仓不在今日预测池里，说明可能停牌、数据缺失或股票池过滤掉了。
    # 为了保守处理，把它排到最低分，优先卖出。
    merged["pred"] = merged["pred"].fillna(-1e9)
    merged["rank"] = merged["rank"].fillna(10**9).astype(int)
    merged["signal_date"] = merged["signal_date"].fillna("")

    return merged


def make_initial_plan(
    candidates: pd.DataFrame,
    topk: int,
    trade_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buy = candidates.head(topk).copy()
    buy["trade_date"] = trade_date
    buy["target_weight"] = 1.0 / topk
    buy["action"] = "buy"
    buy["reason"] = "initial_build_position_topk"

    sell = pd.DataFrame(columns=buy.columns)
    hold = pd.DataFrame(columns=buy.columns)

    plan = pd.concat([sell, buy, hold], ignore_index=True)
    return buy, sell, hold, format_plan(plan)


def make_rebalance_plan(
    candidates: pd.DataFrame,
    positions: pd.DataFrame,
    topk: int,
    dropk: int,
    trade_date: str,
    keep_missing_position: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if topk <= 0:
        raise ValueError("topk must be positive")
    if dropk <= 0:
        raise ValueError("dropk must be positive")
    if dropk > topk:
        raise ValueError("dropk should not be larger than topk")

    current_codes = set(positions["ts_code"].astype(str).tolist())

    pos_pred = attach_prediction_to_positions(positions, candidates)

    # 当前持仓超过 TopK 时，先压回 TopK，避免越持越多。
    max_sell_num = max(dropk, max(0, len(current_codes) - topk + dropk))

    # 默认逻辑：当前持仓中分数最低的 DropK 卖出。
    # 如果某些持仓今日没有预测分数，会被填成极低分，默认优先卖出。
    if keep_missing_position:
        # 如果不想因为缺预测就卖出，可以把缺失项排到最后处理。
        missing_mask = pos_pred["pred"] <= -1e8
        normal_pos = pos_pred[~missing_mask].copy()
        missing_pos = pos_pred[missing_mask].copy()
        sell = normal_pos.sort_values(["pred", "rank"], ascending=[True, False]).head(max_sell_num).copy()
        if len(sell) < max_sell_num:
            extra = missing_pos.head(max_sell_num - len(sell)).copy()
            sell = pd.concat([sell, extra], ignore_index=True)
    else:
        sell = pos_pred.sort_values(["pred", "rank"], ascending=[True, False]).head(max_sell_num).copy()

    sell_codes = set(sell["ts_code"].astype(str).tolist())

    remain_codes = current_codes - sell_codes

    # 目标买入数量：卖几只，买几只；如果当前持仓不足 TopK，则补足。
    need_buy = topk - len(remain_codes)
    need_buy = max(0, need_buy)

    buy = candidates[
        ~candidates["ts_code"].astype(str).isin(remain_codes)
        & ~candidates["ts_code"].astype(str).isin(sell_codes)
    ].head(need_buy).copy()

    buy_codes = set(buy["ts_code"].astype(str).tolist())

    hold = candidates[candidates["ts_code"].astype(str).isin(remain_codes)].copy()

    # 如果有持仓不在 candidates 里，但没被卖出，也要保留在 hold_list 里。
    missing_hold_codes = remain_codes - set(hold["ts_code"].astype(str).tolist())
    if missing_hold_codes:
        missing_hold = positions[positions["ts_code"].astype(str).isin(missing_hold_codes)].copy()
        missing_hold["pred"] = np.nan
        missing_hold["rank"] = np.nan
        missing_hold["signal_date"] = ""
        hold = pd.concat([hold, missing_hold], ignore_index=True)

    # 添加统一字段
    sell["trade_date"] = trade_date
    sell["target_weight"] = 0.0
    sell["action"] = "sell"
    sell["reason"] = "holding_low_score_dropk"

    buy["trade_date"] = trade_date
    buy["target_weight"] = 1.0 / topk
    buy["action"] = "buy"
    buy["reason"] = "new_top_candidate"

    hold["trade_date"] = trade_date
    hold["target_weight"] = 1.0 / topk
    hold["action"] = "hold"
    hold["reason"] = "existing_position_kept"

    plan = pd.concat([sell, buy, hold], ignore_index=True)
    plan = format_plan(plan)

    return format_plan(buy), format_plan(sell), format_plan(hold), plan


def format_plan(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "trade_date",
        "signal_date",
        "ts_code",
        "pred",
        "rank",
        "target_weight",
        "action",
        "reason",
        "buy_date",
        "shares",
        "weight",
    ]

    for c in preferred_cols:
        if c not in df.columns:
            df[c] = ""

    df = df[preferred_cols].copy()

    action_order = {"sell": 0, "buy": 1, "hold": 2}
    df["_action_order"] = df["action"].map(action_order).fillna(9)

    df["_rank_sort"] = pd.to_numeric(df["rank"], errors="coerce").fillna(10**9)
    df = df.sort_values(["_action_order", "_rank_sort", "ts_code"]).drop(
        columns=["_action_order", "_rank_sort"]
    )

    return df.reset_index(drop=True)


def save_outputs(
    buy: pd.DataFrame,
    sell: pd.DataFrame,
    hold: pd.DataFrame,
    plan: pd.DataFrame,
    out_dir: Path,
    trade_date: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = trade_date.replace("-", "")

    paths = {
        "buy": out_dir / f"buy_list_{date_tag}.csv",
        "sell": out_dir / f"sell_list_{date_tag}.csv",
        "hold": out_dir / f"hold_list_{date_tag}.csv",
        "plan": out_dir / f"trade_plan_{date_tag}.csv",
        "latest_buy": out_dir / "latest_buy_list.csv",
        "latest_sell": out_dir / "latest_sell_list.csv",
        "latest_hold": out_dir / "latest_hold_list.csv",
        "latest_plan": out_dir / "latest_trade_plan.csv",
    }

    buy.to_csv(paths["buy"], index=False, encoding="utf-8-sig")
    sell.to_csv(paths["sell"], index=False, encoding="utf-8-sig")
    hold.to_csv(paths["hold"], index=False, encoding="utf-8-sig")
    plan.to_csv(paths["plan"], index=False, encoding="utf-8-sig")

    buy.to_csv(paths["latest_buy"], index=False, encoding="utf-8-sig")
    sell.to_csv(paths["latest_sell"], index=False, encoding="utf-8-sig")
    hold.to_csv(paths["latest_hold"], index=False, encoding="utf-8-sig")
    plan.to_csv(paths["latest_plan"], index=False, encoding="utf-8-sig")

    return paths


def print_summary(buy: pd.DataFrame, sell: pd.DataFrame, hold: pd.DataFrame, plan: pd.DataFrame):
    print("=" * 90)
    print("Trade plan summary")
    print("=" * 90)
    print(f"sell: {len(sell)}")
    print(f"buy : {len(buy)}")
    print(f"hold: {len(hold)}")
    print(f"plan: {len(plan)}")
    print("-" * 90)

    show_cols = ["action", "ts_code", "pred", "rank", "target_weight", "reason"]
    show_cols = [c for c in show_cols if c in plan.columns]

    if len(plan) > 0:
        print(plan[show_cols].to_string(index=False))
    else:
        print("Empty plan.")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        type=str,
        default="outputs_daily/latest_candidates.csv",
        help="最新预测分数文件，至少包含 ts_code,pred，可选 rank,signal_date",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="data/current_positions.csv",
        help="当前持仓文件，至少包含 ts_code，可选 buy_date,shares,weight",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_trade_plan",
        help="输出目录",
    )
    parser.add_argument(
        "--trade_date",
        type=str,
        default=None,
        help="计划交易日期，例如 2026-06-02。不传则使用当前日期。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="目标持仓股票数量",
    )
    parser.add_argument(
        "--dropk",
        type=int,
        default=2,
        help="每日卖出/买入数量",
    )
    parser.add_argument(
        "--keep_missing_position",
        action="store_true",
        help="如果当前持仓股票在预测文件中缺失，则尽量不优先卖出。默认缺失预测会优先卖出。",
    )

    args = parser.parse_args()

    pred_path = Path(args.pred)
    positions_path = Path(args.positions)
    out_dir = Path(args.out_dir)
    trade_date = normalize_trade_date(args.trade_date)

    candidates = read_candidates(pred_path)
    positions = read_positions(positions_path)

    if len(positions) == 0:
        buy, sell, hold, plan = make_initial_plan(
            candidates=candidates,
            topk=args.topk,
            trade_date=trade_date,
        )
    else:
        buy, sell, hold, plan = make_rebalance_plan(
            candidates=candidates,
            positions=positions,
            topk=args.topk,
            dropk=args.dropk,
            trade_date=trade_date,
            keep_missing_position=args.keep_missing_position,
        )

    paths = save_outputs(
        buy=buy,
        sell=sell,
        hold=hold,
        plan=plan,
        out_dir=out_dir,
        trade_date=trade_date,
    )

    print_summary(buy, sell, hold, plan)

    print("Saved files:")
    for k, v in paths.items():
        print(f"{k:>12}: {v}")


if __name__ == "__main__":
    main()