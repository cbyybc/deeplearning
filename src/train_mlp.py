import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from data_utils import load_config, prepare_data, save_json, set_seed
from metrics import calc_metrics
from model import MLPRegressor


def make_loader(df, feature_cols, label_col, batch_size, shuffle):
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[label_col].values, dtype=torch.float32)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_loss_and_pred(model, loader, device, loss_fn):
    model.eval()
    losses = []
    preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item() * len(x))
            preds.append(pred.detach().cpu().numpy())

    avg_loss = np.sum(losses) / len(loader.dataset)
    preds = np.concatenate(preds)

    return avg_loss, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    set_seed(cfg["train"]["seed"])

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]

    print("=" * 80)
    print("准备数据")
    print("=" * 80)

    train_df, valid_df, backtest_df, preprocess_state = prepare_data(cfg)

    save_json(preprocess_state, os.path.join(output_dir, "preprocess_state.json"))

    train_loader = make_loader(
        train_df, feature_cols, label_col,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True
    )

    valid_loader = make_loader(
        valid_df, feature_cols, label_col,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False
    )

    print("=" * 80)
    print("初始化模型")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = MLPRegressor(
        input_dim=len(feature_cols),
        hidden_dims=cfg["train"]["hidden_dims"],
        dropout=cfg["train"]["dropout"],
    ).to(device)

    if cfg["train"].get("loss", "smooth_l1") == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_valid_loss = float("inf")
    best_epoch = -1
    patience = cfg["train"]["early_stop_patience"]
    bad_count = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
    }

    print("=" * 80)
    print("开始训练")
    print("=" * 80)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(train_losses))

        train_loss = float(np.mean(train_losses))
        valid_loss, valid_pred = evaluate_loss_and_pred(model, valid_loader, device, loss_fn)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(float(valid_loss))

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            bad_count = 0

            ckpt = {
                "model_state_dict": model.state_dict(),
                "input_dim": len(feature_cols),
                "feature_cols": feature_cols,
                "label_col": label_col,
                "config": cfg,
            }
            torch.save(ckpt, os.path.join(output_dir, "models", "best_mlp.pt"))
            print(f"  -> 保存最佳模型，valid_loss={best_valid_loss:.6f}")
        else:
            bad_count += 1
            if bad_count >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
                break

    # 保存 loss 曲线
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["valid_loss"], label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("MLP Training Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", "loss_curve.png"), dpi=200)
    plt.close()

    # 加载最佳模型，输出验证集预测和指标
    ckpt = torch.load(os.path.join(output_dir, "models", "best_mlp.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    valid_loss, valid_pred = evaluate_loss_and_pred(model, valid_loader, device, loss_fn)

    pred_df = valid_df[["ts_code", "trade_date", label_col]].copy()
    pred_df = pred_df.rename(columns={label_col: "label"})
    pred_df["pred"] = valid_pred

    pred_path = os.path.join(output_dir, "predictions", "valid_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = calc_metrics(pred_df, pred_col="pred", label_col="label")
    metrics["valid_loss"] = float(valid_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_valid_loss"] = float(best_valid_loss)

    save_json(metrics, os.path.join(output_dir, "metrics_valid.json"))

    print("=" * 80)
    print("验证集指标")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print("训练完成")
    print(f"最佳模型: {os.path.join(output_dir, 'models', 'best_mlp.pt')}")
    print(f"验证集预测: {pred_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
