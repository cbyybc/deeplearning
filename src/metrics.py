from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calc_daily_ic(pred_df: pd.DataFrame, pred_col="pred", label_col="label") -> pd.Series:
    def _corr(x):
        if x[pred_col].nunique() <= 1 or x[label_col].nunique() <= 1:
            return np.nan
        return x[pred_col].corr(x[label_col])

    return pred_df.groupby("trade_date").apply(_corr).dropna()


def calc_daily_rank_ic(pred_df: pd.DataFrame, pred_col="pred", label_col="label") -> pd.Series:
    def _rank_corr(x):
        if x[pred_col].nunique() <= 1 or x[label_col].nunique() <= 1:
            return np.nan
        return x[pred_col].corr(x[label_col], method="spearman")

    return pred_df.groupby("trade_date").apply(_rank_corr).dropna()


def calc_metrics(pred_df: pd.DataFrame, pred_col="pred", label_col="label") -> Dict:
    ic = calc_daily_ic(pred_df, pred_col, label_col)
    rank_ic = calc_daily_rank_ic(pred_df, pred_col, label_col)

    direction_acc = ((pred_df[pred_col] > 0) == (pred_df[label_col] > 0)).mean()

    metrics = {
        "IC_mean": float(ic.mean()) if len(ic) else np.nan,
        "IC_std": float(ic.std()) if len(ic) else np.nan,
        "ICIR": float(ic.mean() / ic.std()) if len(ic) and ic.std() != 0 else np.nan,
        "RankIC_mean": float(rank_ic.mean()) if len(rank_ic) else np.nan,
        "RankIC_std": float(rank_ic.std()) if len(rank_ic) else np.nan,
        "RankICIR": float(rank_ic.mean() / rank_ic.std()) if len(rank_ic) and rank_ic.std() != 0 else np.nan,
        "DirectionAcc": float(direction_acc),
        "num_days": int(pred_df["trade_date"].nunique()),
        "num_samples": int(len(pred_df)),
    }

    return metrics


def calc_backtest_metrics(nav_df: pd.DataFrame) -> Dict:
    nav = nav_df["nav"].astype(float)
    daily_ret = nav.pct_change().fillna(0)

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    num_days = max(len(nav_df), 1)
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / num_days) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else np.nan

    cummax = nav.cummax()
    drawdown = nav / cummax - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "num_days": int(num_days),
    }
