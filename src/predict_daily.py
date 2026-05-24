import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_lstm import LSTMRegressor


FEATURE_COLS = [
    "close_rank",
    "vol_rank",
    "amount_rank",
    "amplitude",
    "body_ratio",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "rel_ma5",
    "rel_ma10",
    "rel_ma20",
    "rsi_14",
    "macd_dif",
    "macd_dea",
    "macd_bar",
    "vol_std_10",
    "vol_std_20",
    "vol_chg",
    "amount_chg",
]

MARKET_DATA_COLS = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
]

NAME_COLS = ["name", "stock_name", "security_name", "display_name", "股票名称", "名称"]
EXCHANGE_COLS = ["exchange", "market", "board", "list_board", "交易所", "市场", "板块"]
ST_FLAG_COLS = ["is_st", "st", "isST", "special_treatment", "risk_warning", "是否ST"]
STATUS_COLS = ["list_status", "status", "trade_status", "上市状态", "交易状态"]
BASIC_REQUIRED_COLS = ["ts_code", "name", "market"]
ST_REQUIRED_COLS = ["ts_code", "trade_date"]


def read_all_stock_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(list(data_dir.rglob("*.parquet")) + list(data_dir.rglob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No csv/parquet files found in {data_dir}")

    dfs = []
    for p in files:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        if "ts_code" not in df.columns:
            df["ts_code"] = p.stem

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    missing_cols = [col for col in MARKET_DATA_COLS if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Market data in {data_dir} is missing required columns: {missing_cols}. "
            "Use daily OHLCV files, not stock-status or stock-pool files."
        )

    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str))
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return data


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _truthy_flag(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "是", "st"})


def _bad_status(s: pd.Series) -> pd.Series:
    text = s.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"d", "delist", "delisted", "退市", "暂停上市", "终止上市", "停牌"})


def read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path)

    return table


def read_basic_csv(path: Path) -> pd.DataFrame:
    basic = read_table(path, "basic.csv")
    missing = [col for col in BASIC_REQUIRED_COLS if col not in basic.columns]
    if missing:
        raise ValueError(f"basic.csv is missing required columns {missing}: {path}")

    basic = basic.copy()
    basic["ts_code"] = basic["ts_code"].astype(str)
    return basic.drop_duplicates("ts_code", keep="last")


def read_stock_st_file(path: Path, signal_date: pd.Timestamp) -> pd.DataFrame:
    stock_st = read_table(path, "stock_st")
    missing = [col for col in ST_REQUIRED_COLS if col not in stock_st.columns]
    if missing:
        raise ValueError(f"stock_st file is missing required columns {missing}: {path}")

    stock_st = stock_st.copy()
    stock_st["ts_code"] = stock_st["ts_code"].astype(str)
    stock_st["trade_date"] = pd.to_datetime(stock_st["trade_date"].astype(str))
    stock_st = stock_st[stock_st["trade_date"] <= signal_date]
    if stock_st.empty:
        print(f"stock_st filter date: no ST rows on or before {signal_date.date()}")
        return stock_st

    effective_date = stock_st["trade_date"].max()
    stock_st = stock_st[stock_st["trade_date"] == effective_date]
    print(f"stock_st filter date: using {effective_date.date()} for signal_date {signal_date.date()}")
    return stock_st.drop_duplicates("ts_code", keep="last")


def build_required_stock_meta(
    basic_csv: Path,
    stock_st_file: Path,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    basic = read_basic_csv(basic_csv)
    stock_st = read_stock_st_file(stock_st_file, signal_date)

    meta = basic.copy()
    st_codes = set(stock_st["ts_code"].astype(str))
    meta["is_st"] = meta["ts_code"].astype(str).isin(st_codes).astype(int)
    return meta


def attach_stock_meta(df: pd.DataFrame, stock_meta: pd.DataFrame | None) -> pd.DataFrame:
    if stock_meta is None:
        return df

    keep_cols = ["ts_code"]
    for cols in [NAME_COLS, EXCHANGE_COLS, ST_FLAG_COLS, STATUS_COLS]:
        for col in cols:
            if col in stock_meta.columns and col not in keep_cols:
                keep_cols.append(col)

    if keep_cols == ["ts_code"]:
        return df

    merged = df.merge(stock_meta[keep_cols], on="ts_code", how="left", suffixes=("", "_meta"))
    for col in keep_cols:
        if col == "ts_code":
            continue
        meta_col = f"{col}_meta"
        if meta_col not in merged.columns:
            continue
        if col in merged.columns:
            base = merged[col]
            fill_mask = base.isna() | (base.astype(str).str.strip() == "")
            merged.loc[fill_mask, col] = merged.loc[fill_mask, meta_col]
            merged = merged.drop(columns=[meta_col])
        else:
            merged = merged.rename(columns={meta_col: col})

    return merged


def filter_stock_pool(
    df: pd.DataFrame,
    stock_meta: pd.DataFrame,
) -> pd.DataFrame:
    df = attach_stock_meta(df, stock_meta)
    before_rows = len(df)
    before_stocks = int(df["ts_code"].nunique())
    codes = df["ts_code"].astype(str)

    # 过滤北交所：常见代码后缀 .BJ，也兼容带交易所/板块字段的股票状态表。
    drop_mask = codes.str.endswith(".BJ")
    exchange_col = _first_existing_col(df, EXCHANGE_COLS)
    if exchange_col is not None:
        exchange_text = df[exchange_col].fillna("").astype(str)
        drop_mask |= exchange_text.str.contains("BJ|BSE|北交|北京证券", case=False, regex=True)

    name_col = _first_existing_col(df, NAME_COLS)
    if name_col is not None:
        name = df[name_col].fillna("").astype(str)
        drop_mask |= name.str.contains(r"ST|\*ST|退市", case=False, na=False, regex=True)

    has_flag_col = any(col in df.columns for col in ST_FLAG_COLS)
    has_status_col = any(col in df.columns for col in STATUS_COLS)
    if name_col is None or not has_flag_col:
        raise ValueError(
            "Cannot guarantee ST stock filtering. Prediction now requires --basic_csv "
            "with ts_code/name/market and --stock_st with ts_code/trade_date so ST names, "
            "the daily ST list, and Beijing Stock Exchange stocks can be filtered."
        )

    for flag_col in ST_FLAG_COLS:
        if flag_col in df.columns:
            drop_mask |= _truthy_flag(df[flag_col])

    for status_col in STATUS_COLS:
        if status_col in df.columns:
            drop_mask |= _bad_status(df[status_col])

    df = df[~drop_mask].copy()
    after_stocks = int(df["ts_code"].nunique())
    print(
        "stock pool filter: "
        f"rows {before_rows}->{len(df)}, stocks {before_stocks}->{after_stocks}, "
        f"removed_stocks={before_stocks - after_stocks}"
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).copy()

    g = df.groupby("ts_code", group_keys=False)

    df["ret_1"] = g["close"].pct_change(1)
    df["ret_5"] = g["close"].pct_change(5)
    df["ret_10"] = g["close"].pct_change(10)
    df["ret_20"] = g["close"].pct_change(20)

    ma5 = g["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    ma10 = g["close"].transform(lambda x: x.rolling(10, min_periods=10).mean())
    ma20 = g["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())

    df["rel_ma5"] = df["close"] / ma5 - 1
    df["rel_ma10"] = df["close"] / ma10 - 1
    df["rel_ma20"] = df["close"] / ma20 - 1

    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    df["vol_chg"] = g["vol"].pct_change(1)
    df["amount_chg"] = g["amount"].pct_change(1)
    df["vol_std_10"] = g["ret_1"].transform(lambda x: x.rolling(10, min_periods=10).std())
    df["vol_std_20"] = g["ret_1"].transform(lambda x: x.rolling(20, min_periods=20).std())

    # RSI
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["ts_code"]).transform(lambda x: x.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["ts_code"]).transform(lambda x: x.rolling(14, min_periods=14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df.groupby("ts_code")["macd_dif"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["macd_bar"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # 每日横截面排名特征
    day_g = df.groupby("trade_date")
    df["close_rank"] = day_g["close"].rank(pct=True)
    df["vol_rank"] = day_g["vol"].rank(pct=True)
    df["amount_rank"] = day_g["amount"].rank(pct=True)

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return df


def load_scaler(scaler_path: Path):
    with open(scaler_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mean = pd.Series(obj["mean"])
    std = pd.Series(obj["std"]).replace(0, 1.0)
    return mean, std


def apply_train_scaler(df: pd.DataFrame, scaler_path: Path) -> pd.DataFrame:
    mean, std = load_scaler(scaler_path)
    out = df.copy()
    out[FEATURE_COLS] = (out[FEATURE_COLS] - mean[FEATURE_COLS]) / std[FEATURE_COLS]
    out[FEATURE_COLS] = out[FEATURE_COLS].clip(-5, 5)
    return out


def build_latest_sequences(df: pd.DataFrame, signal_date, seq_len: int):
    xs = []
    rows = []
    stock_count = int(df["ts_code"].nunique())
    signal_rows = df[df["trade_date"] == signal_date]
    signal_stock_count = int(signal_rows["ts_code"].nunique())
    enough_feature_history = 0
    signal_window_count = 0

    for ts_code, sub in df.groupby("ts_code"):
        sub = sub[sub["trade_date"] <= signal_date].sort_values("trade_date")
        sub = sub.dropna(subset=FEATURE_COLS)

        if len(sub) < seq_len:
            continue
        enough_feature_history += 1

        last = sub.tail(seq_len)
        if last["trade_date"].iloc[-1] != signal_date:
            continue
        signal_window_count += 1

        x = last[FEATURE_COLS].to_numpy(dtype=np.float32)
        row = {
            "signal_date": signal_date,
            "ts_code": ts_code,
            "last_close": last["close"].iloc[-1],
        }
        name_col = _first_existing_col(last, NAME_COLS)
        if name_col is not None:
            row["name"] = last[name_col].dropna().astype(str).iloc[-1] if last[name_col].notna().any() else ""
        xs.append(x)
        rows.append(row)

    if not xs:
        date_text = pd.to_datetime(signal_date).strftime("%Y-%m-%d")
        unique_dates = int(df["trade_date"].nunique())
        raise RuntimeError(
            "No valid sequence generated. "
            f"signal_date={date_text}, seq_len={seq_len}, stocks={stock_count}, "
            f"trade_dates={unique_dates}, stocks_on_signal_date={signal_stock_count}, "
            f"stocks_with_{seq_len}_valid_feature_rows={enough_feature_history}, "
            f"stocks_with_signal_window={signal_window_count}. "
            "Provide OHLCV history through the signal date; rolling features need "
            "roughly 30+ trading days per stock before the final sequence is built."
        )

    return np.stack(xs), pd.DataFrame(rows)


def load_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    cfg_train = ckpt.get("config", {}).get("train", {}) if isinstance(ckpt, dict) else {}
    input_dim = ckpt.get("input_dim", len(FEATURE_COLS)) if isinstance(ckpt, dict) else len(FEATURE_COLS)
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_size=cfg_train.get("hidden_size", 128),
        num_layers=cfg_train.get("num_layers", 2),
        dropout=cfg_train.get("dropout", 0.2),
        bidirectional=cfg_train.get("bidirectional", False),
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, x: np.ndarray, device: str, batch_size: int = 8192):
    preds = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        yb = model(xb).detach().cpu().numpy()
        preds.append(yb)
    return np.concatenate(preds)


def make_trade_plan(
    candidates: pd.DataFrame,
    positions_path: Path,
    trade_date,
    top_k_hold: int = 10,
    drop_k: int = 2,
):
    if positions_path.exists():
        pos = pd.read_csv(positions_path)
    else:
        pos = pd.DataFrame(columns=["ts_code", "buy_date", "shares", "weight"])

    current_hold = set(pos["ts_code"].astype(str).tolist())
    candidates["ts_code"] = candidates["ts_code"].astype(str)

    top_pool = candidates.head(max(top_k_hold * 3, 30)).copy()
    current_df = candidates[candidates["ts_code"].isin(current_hold)].copy()

    rows = []

    if len(current_hold) == 0:
        buy_df = candidates.head(top_k_hold).copy()
        for _, r in buy_df.iterrows():
            rows.append({
                "trade_date": trade_date,
                "signal_date": r["signal_date"],
                "ts_code": r["ts_code"],
                "pred": r["pred"],
                "rank": r["rank"],
                "target_weight": 1.0 / top_k_hold,
                "action": "buy",
                "reason": "initial_top_rank",
            })
        return pd.DataFrame(rows)

    # 卖出：当前持仓中分数最低的 drop_k 只
    sell_df = current_df.sort_values("pred", ascending=True).head(drop_k)
    sell_set = set(sell_df["ts_code"].tolist())

    for _, r in sell_df.iterrows():
        rows.append({
            "trade_date": trade_date,
            "signal_date": r["signal_date"],
            "ts_code": r["ts_code"],
            "pred": r["pred"],
            "rank": r["rank"],
            "target_weight": 0.0,
            "action": "sell",
            "reason": "holding_low_score_drop",
        })

    remain_hold = current_hold - sell_set

    # 买入：从最高分里选未持有的 drop_k 只
    buy_df = top_pool[
        ~top_pool["ts_code"].isin(current_hold)
    ].head(drop_k)

    for _, r in buy_df.iterrows():
        rows.append({
            "trade_date": trade_date,
            "signal_date": r["signal_date"],
            "ts_code": r["ts_code"],
            "pred": r["pred"],
            "rank": r["rank"],
            "target_weight": 1.0 / top_k_hold,
            "action": "buy",
            "reason": "top_rank_new_candidate",
        })

    # 持有
    hold_df = candidates[candidates["ts_code"].isin(remain_hold)]
    for _, r in hold_df.iterrows():
        rows.append({
            "trade_date": trade_date,
            "signal_date": r["signal_date"],
            "ts_code": r["ts_code"],
            "pred": r["pred"],
            "rank": r["rank"],
            "target_weight": 1.0 / top_k_hold,
            "action": "hold",
            "reason": "already_holding_not_dropped",
        })

    plan = pd.DataFrame(rows)
    action_order = {"sell": 0, "buy": 1, "hold": 2}
    plan["action_order"] = plan["action"].map(action_order)
    plan = plan.sort_values(["action_order", "rank"]).drop(columns=["action_order"])
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scaler", type=str, required=True)
    parser.add_argument("--positions", type=str, default="data/current_positions.csv")
    parser.add_argument(
        "--basic_csv",
        type=str,
        required=True,
        help="必填 basic.csv，需含 ts_code/name/market，用于识别股票名称和北交所股票。",
    )
    parser.add_argument(
        "--stock_st",
        type=str,
        required=True,
        help="必填每日 stock_st 文件，需含 ts_code/trade_date，用于按 signal_date 过滤 ST 股票。",
    )
    parser.add_argument("--out_dir", type=str, default="outputs_daily")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--top_k_hold", type=int, default=10)
    parser.add_argument("--drop_k", type=int, default=2)
    parser.add_argument("--signal_date", type=str, default=None)
    parser.add_argument("--trade_date", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    df = read_all_stock_data(data_dir)
    if args.signal_date is None:
        signal_date = df["trade_date"].max()
    else:
        signal_date = pd.to_datetime(args.signal_date)

    stock_meta = build_required_stock_meta(
        basic_csv=Path(args.basic_csv),
        stock_st_file=Path(args.stock_st),
        signal_date=signal_date,
    )
    df = filter_stock_pool(df, stock_meta=stock_meta)
    df = add_features(df)
    df = apply_train_scaler(df, Path(args.scaler))

    if args.trade_date is None:
        # 注意：这里默认只是写成下一天。
        # 正式比赛时，如果中间有周末/节假日，建议手动传入 --trade_date。
        trade_date = signal_date + pd.Timedelta(days=1)
    else:
        trade_date = pd.to_datetime(args.trade_date)

    x, meta = build_latest_sequences(df, signal_date, args.seq_len)

    model = load_model(Path(args.checkpoint), device)
    pred = predict(model, x, device)

    candidates = meta.copy()
    candidates["pred"] = pred
    candidates = candidates.sort_values("pred", ascending=False).reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["signal_date"] = pd.to_datetime(candidates["signal_date"]).dt.strftime("%Y-%m-%d")

    signal_str = pd.to_datetime(signal_date).strftime("%Y%m%d")
    trade_str = pd.to_datetime(trade_date).strftime("%Y%m%d")

    cand_path = out_dir / f"daily_candidates_{signal_str}.csv"
    latest_cand_path = out_dir / "latest_candidates.csv"
    candidates.to_csv(cand_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(latest_cand_path, index=False, encoding="utf-8-sig")

    plan = make_trade_plan(
        candidates=candidates,
        positions_path=Path(args.positions),
        trade_date=pd.to_datetime(trade_date).strftime("%Y-%m-%d"),
        top_k_hold=args.top_k_hold,
        drop_k=args.drop_k,
    )

    plan_path = out_dir / f"trade_plan_{signal_str}_for_{trade_str}.csv"
    latest_plan_path = out_dir / "latest_trade_plan.csv"
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    plan.to_csv(latest_plan_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"signal_date: {pd.to_datetime(signal_date).strftime('%Y-%m-%d')}")
    print(f"trade_date : {pd.to_datetime(trade_date).strftime('%Y-%m-%d')}")
    print(f"candidates : {cand_path}")
    print(f"trade plan : {plan_path}")
    print("=" * 80)
    print(plan[["action", "ts_code", "pred", "rank", "target_weight", "reason"]])


if __name__ == "__main__":
    main()
