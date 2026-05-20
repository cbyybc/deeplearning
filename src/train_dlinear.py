import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from data_utils import load_config, prepare_data, save_json, set_seed
from metrics import calc_metrics
from sequence_dataset import StockSequenceDataset
from model_dlinear import DLinearRegressor


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    pbar = tqdm(loader, desc="train", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=float(np.mean(losses)))

    return float(np.mean(losses))


def evaluate(model, loader, loss_fn, device, return_pred=False):
    model.eval()
    loss_sum = 0.0
    preds, labels, codes, dates = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            if len(batch) == 4:
                x, y, ts_code, trade_date = batch
            else:
                x, y = batch
                ts_code, trade_date = None, None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)
            loss_sum += loss.item() * len(x)

            if return_pred:
                preds.append(pred.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
                codes.extend(list(ts_code))
                dates.extend([int(d) for d in trade_date])

    avg_loss = float(loss_sum / len(loader.dataset))

    if return_pred:
        pred_df = pd.DataFrame({
            "ts_code": codes,
            "trade_date": dates,
            "label": np.concatenate(labels),
            "pred": np.concatenate(preds),
        })
        return avg_loss, pred_df

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_dlinear_seq10_e60.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]
    seq_len = cfg["sequence"]["seq_len"]
    moving_avg = cfg["sequence"].get("moving_avg", 3)

    print("=" * 80)
    print("准备 DLinear 数据")
    print("=" * 80)

    train_df, valid_df, _, preprocess_state = prepare_data(cfg)
    preprocess_state["seq_len"] = seq_len
    preprocess_state["moving_avg"] = moving_avg
    save_json(preprocess_state, os.path.join(out_dir, "preprocess_state_dlinear.json"))

    train_ds = StockSequenceDataset(train_df, feature_cols, label_col, seq_len=seq_len, return_meta=False)
    valid_ds = StockSequenceDataset(valid_df, feature_cols, label_col, seq_len=seq_len, return_meta=True)

    train_loader = make_loader(train_ds, cfg["train"]["batch_size"], True, cfg["train"].get("num_workers", 0))
    valid_loader = make_loader(valid_ds, cfg["train"]["batch_size"], False, cfg["train"].get("num_workers", 0))

    print("=" * 80)
    print("初始化 DLinear")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = DLinearRegressor(
        seq_len=seq_len,
        feature_dim=len(feature_cols),
        moving_avg=moving_avg,
        dropout=cfg["train"].get("dropout", 0.1),
    ).to(device)

    loss_fn = torch.nn.MSELoss() if cfg["train"].get("loss") == "mse" else torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_valid_loss = float("inf")
    best_epoch = -1
    bad_count = 0
    history = []

    print("=" * 80)
    print("开始训练 DLinear")
    print("=" * 80)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device, return_pred=False)

        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            bad_count = 0

            ckpt = {
                "model_state_dict": model.state_dict(),
                "seq_len": seq_len,
                "feature_dim": len(feature_cols),
                "feature_cols": feature_cols,
                "label_col": label_col,
                "moving_avg": moving_avg,
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(out_dir, "models", "best_dlinear.pt"))
            print(f"  -> 保存最佳 DLinear, valid_loss={best_valid_loss:.6f}")
        else:
            bad_count += 1
            if bad_count >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "training_history_dlinear.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["valid_loss"], label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DLinear Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "loss_curve_dlinear.png"), dpi=200)
    plt.close()

    ckpt = torch.load(os.path.join(out_dir, "models", "best_dlinear.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    valid_loss, valid_pred_df = evaluate(model, valid_loader, loss_fn, device, return_pred=True)
    valid_pred_df.to_csv(os.path.join(out_dir, "predictions", "valid_predictions_dlinear.csv"), index=False, encoding="utf-8-sig")

    metrics = calc_metrics(valid_pred_df, pred_col="pred", label_col="label")
    metrics["valid_loss"] = float(valid_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_valid_loss"] = float(best_valid_loss)
    save_json(metrics, os.path.join(out_dir, "metrics_valid_dlinear.json"))

    print("=" * 80)
    print("DLinear 验证集指标")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("DLinear 训练完成")
    print(f"best model: {os.path.join(out_dir, 'models', 'best_dlinear.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
