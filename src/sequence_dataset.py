import numpy as np
import torch
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """
    按 ts_code 构造滑动窗口序列样本。

    第 i 个样本:
        X = 当前股票 [i-seq_len+1, ..., i] 的历史特征序列
        y = 当前股票第 i 行的标签

    对应交易含义:
        t 日收盘后，使用 t 日及以前 seq_len 天特征，
        预测 t+1 日开盘到收盘收益 label_oc_1d。
    """

    def __init__(self, df, feature_cols, label_col, seq_len=20, return_meta=False):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len
        self.return_meta = return_meta

        self.features = []
        self.labels = []
        self.codes = []
        self.dates = []
        self.index = []

        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        sid = 0
        for ts_code, g in df.groupby("ts_code", sort=False):
            g = g.sort_values("trade_date").reset_index(drop=True)
            if len(g) < seq_len:
                continue

            x = g[feature_cols].values.astype("float32")
            y = g[label_col].values.astype("float32")
            d = g["trade_date"].values.astype("int64")

            self.features.append(x)
            self.labels.append(y)
            self.codes.append(ts_code)
            self.dates.append(d)

            for end_idx in range(seq_len - 1, len(g)):
                self.index.append((sid, end_idx))

            sid += 1

        print(
            f"StockSequenceDataset: stocks={len(self.features)}, "
            f"samples={len(self.index)}, seq_len={seq_len}, "
            f"feature_dim={len(feature_cols)}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, end_idx = self.index[idx]
        start_idx = end_idx - self.seq_len + 1

        x = self.features[sid][start_idx:end_idx + 1]
        y = self.labels[sid][end_idx]

        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)

        if self.return_meta:
            return x, y, self.codes[sid], int(self.dates[sid][end_idx])

        return x, y
