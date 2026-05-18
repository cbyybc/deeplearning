import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from data_utils import load_config, prepare_data, save_json
from metrics import calc_metrics
from model import MLPRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["output_dir"]

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    _, valid_df, _, _ = prepare_data(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(output_dir, "models", "best_mlp.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MLPRegressor(
        input_dim=ckpt["input_dim"],
        hidden_dims=cfg["train"]["hidden_dims"],
        dropout=cfg["train"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.tensor(valid_df[feature_cols].values, dtype=torch.float32)
    y = valid_df[label_col].values

    loader = DataLoader(TensorDataset(x), batch_size=cfg["train"]["batch_size"], shuffle=False)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.extend(pred)

    pred_df = valid_df[["ts_code", "trade_date", label_col]].copy()
    pred_df = pred_df.rename(columns={label_col: "label"})
    pred_df["pred"] = preds

    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    pred_df.to_csv(os.path.join(output_dir, "predictions", "valid_predictions.csv"), index=False, encoding="utf-8-sig")

    metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    save_json(metrics, os.path.join(output_dir, "metrics_valid.json"))

    print("Valid metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
