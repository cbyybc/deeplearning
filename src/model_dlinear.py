import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.permute(0, 2, 1)
        trend = self.avg(x_pad)
        trend = trend.permute(0, 2, 1)
        return trend


class DLinearRegressor(nn.Module):
    """
    DLinear-style stock return predictor.

    Input:
        x: [batch, seq_len, feature_dim]
    Output:
        pred_score: [batch]
    """

    def __init__(self, seq_len: int, feature_dim: int, moving_avg: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.input_norm = nn.LayerNorm(feature_dim)
        self.decomp = MovingAverage(kernel_size=moving_avg)

        self.linear_seasonal = nn.Linear(seq_len, 1)
        self.linear_trend = nn.Linear(seq_len, 1)

        self.dropout = nn.Dropout(dropout)
        self.feature_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        trend = self.decomp(x)
        seasonal = x - trend

        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        seasonal_out = self.linear_seasonal(seasonal).squeeze(-1)
        trend_out = self.linear_trend(trend).squeeze(-1)

        feat = self.dropout(seasonal_out + trend_out)
        return self.feature_head(feat).squeeze(-1)
