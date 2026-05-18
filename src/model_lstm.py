import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    输入:
        x: [batch, seq_len, feature_dim]

    输出:
        pred_score: [batch]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
