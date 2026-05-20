import json
import os
from copy import deepcopy


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


DATA_PATH = "Datasets/processed/all_stock_features_with_t1_labels.parquet"
LABEL_COL = "label_oo_1d"


tasks = [
    {
        "src": "configs/config.json",
        "dst": "configs/config_mlp_oo.json",
        "output_dir": "outputs_mlp_oo",
    },
    {
        "src": "configs/config_lstm_seq10.json",
        "dst": "configs/config_lstm_seq10_oo_e60.json",
        "output_dir": "outputs_lstm_seq10_oo_e60",
    },
    {
        "src": "configs/config_lstm_seq20_e60.json",
        "dst": "configs/config_lstm_seq20_oo_e60.json",
        "output_dir": "outputs_lstm_seq20_oo_e60",
    },
    {
        "src": "configs/config_dlinear_seq10_e60.json",
        "dst": "configs/config_dlinear_seq10_oo_e60.json",
        "output_dir": "outputs_dlinear_seq10_oo_e60",
    },
    {
        "src": "configs/config_dlinear_seq20_e60.json",
        "dst": "configs/config_dlinear_seq20_oo_e60.json",
        "output_dir": "outputs_dlinear_seq20_oo_e60",
    },
]


for task in tasks:
    cfg = load_json(task["src"])
    cfg = deepcopy(cfg)

    cfg["data_path"] = DATA_PATH
    cfg["label_col"] = LABEL_COL
    cfg["output_dir"] = task["output_dir"]

    # 确保 e60 设置
    if "train" in cfg:
        cfg["train"]["epochs"] = 60
        cfg["train"]["early_stop_patience"] = 8

    save_json(cfg, task["dst"])
    print("saved:", task["dst"])
