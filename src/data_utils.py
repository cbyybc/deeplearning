import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    lower = data_path.lower()

    if lower.endswith(".pkl") or lower.endswith(".pickle"):
        df = pd.read_pickle(data_path)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")

    return df


def check_required_columns(df: pd.DataFrame, feature_cols: List[str], label_col: str):
    required = ["ts_code", "trade_date", label_col] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")


def time_split(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = cfg["split"]

    train_df = df[
        (df["trade_date"] >= s["train_start"]) &
        (df["trade_date"] <= s["train_end"])
    ].copy()

    valid_df = df[
        (df["trade_date"] >= s["valid_start"]) &
        (df["trade_date"] <= s["valid_end"])
    ].copy()

    backtest_df = df[
        (df["trade_date"] >= s["backtest_start"]) &
        (df["trade_date"] <= s["backtest_end"])
    ].copy()

    print(f"train shape: {train_df.shape}, date: {train_df['trade_date'].min()} - {train_df['trade_date'].max()}")
    print(f"valid shape: {valid_df.shape}, date: {valid_df['trade_date'].min()} - {valid_df['trade_date'].max()}")
    print(f"backtest shape: {backtest_df.shape}, date: {backtest_df['trade_date'].min()} - {backtest_df['trade_date'].max()}")

    if train_df.empty or valid_df.empty:
        raise ValueError("训练集或验证集为空，请检查时间划分。")

    return train_df, valid_df, backtest_df


def clip_by_train_quantile(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cfg: Dict,
):
    pp = cfg["preprocess"]

    # 特征裁剪
    feature_bounds = {}
    for col in feature_cols:
        lower = train_df[col].quantile(pp["feature_clip_lower"])
        upper = train_df[col].quantile(pp["feature_clip_upper"])
        feature_bounds[col] = (float(lower), float(upper))

        train_df[col] = train_df[col].clip(lower, upper)
        valid_df[col] = valid_df[col].clip(lower, upper)
        if not backtest_df.empty:
            backtest_df[col] = backtest_df[col].clip(lower, upper)

    # 标签裁剪
    label_lower = train_df[label_col].quantile(pp["label_clip_lower"])
    label_upper = train_df[label_col].quantile(pp["label_clip_upper"])

    train_df[label_col] = train_df[label_col].clip(label_lower, label_upper)
    valid_df[label_col] = valid_df[label_col].clip(label_lower, label_upper)
    if not backtest_df.empty:
        backtest_df[label_col] = backtest_df[label_col].clip(label_lower, label_upper)

    label_bounds = (float(label_lower), float(label_upper))

    print(f"label clip range for {label_col}: {label_bounds}")

    return train_df, valid_df, backtest_df, feature_bounds, label_bounds


def standardize_by_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    feature_cols: List[str],
):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)

    train_df[feature_cols] = (train_df[feature_cols] - mean) / std
    valid_df[feature_cols] = (valid_df[feature_cols] - mean) / std
    if not backtest_df.empty:
        backtest_df[feature_cols] = (backtest_df[feature_cols] - mean) / std

    return train_df, valid_df, backtest_df, mean, std


def prepare_data(cfg: Dict):
    data_path = cfg["data_path"]
    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    df = load_feature_data(data_path)
    check_required_columns(df, feature_cols, label_col)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ts_code", "trade_date", label_col] + feature_cols).reset_index(drop=True)

    train_df, valid_df, backtest_df = time_split(df, cfg)

    train_df, valid_df, backtest_df, feature_bounds, label_bounds = clip_by_train_quantile(
        train_df, valid_df, backtest_df, feature_cols, label_col, cfg
    )

    train_df, valid_df, backtest_df, mean, std = standardize_by_train(
        train_df, valid_df, backtest_df, feature_cols
    )

    preprocess_state = {
        "feature_cols": feature_cols,
        "label_col": label_col,
        "feature_bounds": feature_bounds,
        "label_bounds": label_bounds,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }

    return train_df, valid_df, backtest_df, preprocess_state


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
