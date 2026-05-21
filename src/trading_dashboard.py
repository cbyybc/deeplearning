"""
Streamlit 图形化交易计划界面

功能：
1. 上传/选择每日最新行情数据，调用 src/predict_daily.py 生成 latest_candidates.csv。
2. 上传/编辑当前持仓，调用 src/make_trade_plan.py 生成 buy/sell/hold/trade_plan。
3. 在网页中查看买入、卖出、持有清单，并下载 CSV。

运行：
    streamlit run src/trading_dashboard.py

推荐目录：
    project_root/
      src/
        predict_daily.py
        make_trade_plan.py
        trading_dashboard.py
      data/
        current_positions.csv
      outputs_daily/
      outputs_trade_plan/
"""

from __future__ import annotations

import os
import sys
import zipfile
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


APP_TITLE = "深度学习股票模拟交易助手"


def infer_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "src":
        return script_dir.parent
    return script_dir


DEFAULT_PROJECT_ROOT = infer_project_root()


# -----------------------------
# Basic utilities
# -----------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_path(s: str | Path) -> Path:
    return Path(str(s)).expanduser().resolve()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_uploaded_files(files: Iterable, out_dir: Path) -> list[Path]:
    ensure_dir(out_dir)
    saved: list[Path] = []
    for f in files:
        dst = out_dir / f.name
        with open(dst, "wb") as w:
            w.write(f.getbuffer())
        saved.append(dst)

        if dst.suffix.lower() == ".zip":
            extract_dir = out_dir / dst.stem
            ensure_dir(extract_dir)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(extract_dir)
            saved.append(extract_dir)

    return saved


def find_data_dir_from_upload(saved_paths: list[Path], fallback_dir: Path) -> Path:
    # 如果上传了 zip，优先使用解压目录；否则使用保存目录。
    dirs = [p for p in saved_paths if p.is_dir()]
    if dirs:
        return dirs[-1]
    return fallback_dir


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout


def csv_download_button(df: pd.DataFrame, label: str, filename: str):
    if df.empty:
        st.download_button(label, data="", file_name=filename, mime="text/csv", disabled=True)
        return
    data = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(label, data=data, file_name=filename, mime="text/csv")


def show_table(title: str, df: pd.DataFrame, height: int = 360):
    st.subheader(title)
    if df.empty:
        st.info("暂无数据。")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def normalize_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ts_code", "buy_date", "shares", "weight"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "ts_code" not in df.columns:
        st.warning("当前持仓文件缺少 ts_code 列，已返回空持仓。")
        return pd.DataFrame(columns=cols)
    out = df.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols]


def infer_signal_date(candidates: pd.DataFrame) -> str:
    if not candidates.empty and "signal_date" in candidates.columns:
        s = candidates["signal_date"].dropna().astype(str)
        if len(s) > 0 and s.iloc[0] != "":
            return s.iloc[0]
    return ""


def trade_date_after_signal(signal_date_text: str, trade_date_value: date) -> bool | None:
    if not signal_date_text:
        return None
    try:
        return pd.to_datetime(trade_date_value) > pd.to_datetime(signal_date_text)
    except (TypeError, ValueError):
        return None


def candidate_pool_warnings(candidates: pd.DataFrame) -> list[str]:
    warnings = []
    if candidates.empty:
        return warnings
    if "ts_code" not in candidates.columns or "pred" not in candidates.columns:
        warnings.append("候选文件缺少 `ts_code` 或 `pred` 列，交易计划脚本无法可靠生成调仓清单。")
        return warnings

    codes = candidates["ts_code"].astype(str)
    bj_count = int(codes.str.endswith(".BJ").sum())
    if bj_count:
        warnings.append(f"候选清单仍包含 {bj_count} 只 `.BJ` 股票，请先排除北交所股票。")

    if "name" in candidates.columns:
        st_count = int(candidates["name"].astype(str).str.contains("ST", case=False, na=False).sum())
        if st_count:
            warnings.append(f"候选清单仍包含 {st_count} 只名称含 ST 的股票，请先排除。")
    else:
        warnings.append("候选清单没有 `name` 列，界面无法再次核对 ST 股票；请确认上游输入已过滤。")

    return warnings


def file_state_label(path: Path) -> str:
    return "已就绪" if path.exists() else "缺失"


# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

st.title("📈 深度学习股票模拟交易助手")
st.caption("上传最新数据 → 调用每日预测脚本 → 生成 buy/sell/hold 交易计划 → 下载清单用于同花顺模拟盘。")

with st.sidebar:
    st.header("路径与参数")

    project_root = to_path(st.text_input("项目根目录", str(DEFAULT_PROJECT_ROOT)))

    predict_script = to_path(st.text_input("每日预测脚本", str(project_root / "src" / "predict_daily.py")))
    plan_script = to_path(st.text_input("交易计划脚本", str(project_root / "src" / "make_trade_plan.py")))

    positions_path = to_path(st.text_input("当前持仓文件", str(project_root / "data" / "current_positions.csv")))
    outputs_daily = to_path(st.text_input("预测输出目录", str(project_root / "outputs_daily")))
    outputs_trade = to_path(st.text_input("交易计划输出目录", str(project_root / "outputs_trade_plan")))

    st.divider()
    st.subheader("策略参数")
    topk = st.number_input("目标持仓 TopK", min_value=1, max_value=50, value=10, step=1)
    dropk = st.number_input("每日换仓 DropK", min_value=1, max_value=10, value=2, step=1)
    trade_date = st.date_input("计划交易日期", value=date.today())
    keep_missing_position = st.checkbox("持仓缺少预测时尽量不优先卖出", value=False)

    st.divider()
    st.subheader("预测参数")
    data_dir_manual = st.text_input("行情数据目录", str(project_root / "../Datasets/stocks"))
    checkpoint = st.text_input(
        "模型权重 checkpoint",
        str(project_root / "outputs_lstm_seq10_oo_e60" / "models" / "best_lstm.pt"),
    )
    scaler = st.text_input(
        "训练集预处理参数",
        str(project_root / "outputs_lstm_seq10_oo_e60" / "preprocess_state_lstm.json"),
    )
    seq_len = st.number_input("seq_len", min_value=1, max_value=120, value=10, step=1)
    signal_date = st.text_input("signal_date，可留空自动识别最新日期", "")
    device = st.selectbox("device", ["cuda", "cpu"], index=0)


latest_candidates_path = outputs_daily / "latest_candidates.csv"
current_candidates = read_csv_if_exists(latest_candidates_path)
current_positions = normalize_positions_df(read_csv_if_exists(positions_path))
known_signal_date = infer_signal_date(current_candidates)
parameter_ok = int(dropk) <= int(topk)
signal_trade_ok = trade_date_after_signal(known_signal_date, trade_date)

st.subheader("比赛执行预检")
preflight_cols = st.columns(5)
preflight_cols[0].metric("当前持仓数", len(current_positions))
preflight_cols[1].metric("候选股票数", len(current_candidates))
preflight_cols[2].metric("预测脚本", file_state_label(predict_script))
preflight_cols[3].metric("交易脚本", file_state_label(plan_script))
preflight_cols[4].metric("模型输入", "已就绪" if Path(checkpoint).exists() and Path(scaler).exists() else "待核对")

if not parameter_ok:
    st.error("DropK 不能大于 TopK。请先调整策略参数，再生成预测或交易计划。")
if signal_trade_ok is False:
    st.error("计划交易日期必须晚于候选信号日期，才能保持第 t 日盘后出信号、第 t+1 日交易。")
elif known_signal_date:
    st.info(f"当前候选信号日期：{known_signal_date}；计划交易日期：{pd.to_datetime(trade_date).strftime('%Y-%m-%d')}。")
else:
    st.info("尚未识别候选信号日期。生成或上传 latest_candidates.csv 后会在这里检查 T+1 日期关系。")

for warning in candidate_pool_warnings(current_candidates):
    st.warning(warning)


# -----------------------------
# Tabs
# -----------------------------
tab_data, tab_position, tab_predict, tab_plan, tab_view = st.tabs(
    ["1 上传/选择数据", "2 当前持仓", "3 生成预测", "4 生成交易计划", "5 查看与下载"]
)


# -----------------------------
# 1. Upload/select market data
# -----------------------------
with tab_data:
    st.header("1. 上传或选择每日最新行情数据")
    st.write("可以上传 csv/parquet/zip，也可以直接使用左侧填写的行情数据目录。zip 会自动解压。")

    upload_dir = ensure_dir(project_root / "data" / "daily_uploads" / date.today().strftime("%Y%m%d"))
    uploaded = st.file_uploader(
        "上传每日最新数据文件",
        type=["csv", "parquet", "zip"],
        accept_multiple_files=True,
    )

    selected_data_dir = to_path(data_dir_manual)

    if uploaded:
        saved_paths = save_uploaded_files(uploaded, upload_dir)
        selected_data_dir = find_data_dir_from_upload(saved_paths, upload_dir)
        st.success(f"已保存上传文件到：{upload_dir}")
        st.info(f"本次将使用数据目录：{selected_data_dir}")
        st.session_state["selected_data_dir"] = str(selected_data_dir)
        st.write("已保存/解压的路径：")
        for p in saved_paths:
            st.code(str(p))
    else:
        st.session_state.setdefault("selected_data_dir", str(selected_data_dir))
        st.info(f"当前使用左侧行情数据目录：{st.session_state['selected_data_dir']}")

    if Path(st.session_state["selected_data_dir"]).exists():
        files = list(Path(st.session_state["selected_data_dir"]).glob("**/*"))[:20]
        if files:
            st.write("目录预览：")
            st.dataframe(pd.DataFrame({"path": [str(p) for p in files]}), use_container_width=True, height=260)
    else:
        st.warning("当前数据目录不存在。请上传数据或修改左侧行情数据目录。")


# -----------------------------
# 2. Positions editor
# -----------------------------
with tab_position:
    st.header("2. 当前持仓")
    st.write("这里维护你同花顺模拟盘的真实持仓。脚本会基于这些持仓生成卖出、买入和持有清单。")

    ensure_dir(positions_path.parent)

    pos_upload = st.file_uploader("上传 current_positions.csv", type=["csv"], key="pos_upload")
    if pos_upload is not None:
        uploaded_pos = pd.read_csv(pos_upload)
        uploaded_pos = normalize_positions_df(uploaded_pos)
        uploaded_pos.to_csv(positions_path, index=False, encoding="utf-8-sig")
        st.success(f"已更新当前持仓文件：{positions_path}")

    pos_df = normalize_positions_df(read_csv_if_exists(positions_path))

    edited = st.data_editor(
        pos_df,
        num_rows="dynamic",
        use_container_width=True,
        key="positions_editor",
        column_config={
            "ts_code": st.column_config.TextColumn("ts_code", help="股票代码，例如 000001.SZ"),
            "buy_date": st.column_config.TextColumn("buy_date", help="买入日期，可为空"),
            "shares": st.column_config.NumberColumn("shares", help="持股数量，可为空"),
            "weight": st.column_config.NumberColumn("weight", help="仓位比例，可为空，例如 0.1"),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("保存当前持仓", type="primary"):
            edited = normalize_positions_df(edited)
            edited.to_csv(positions_path, index=False, encoding="utf-8-sig")
            st.success(f"已保存：{positions_path}")
    with col_b:
        if st.button("创建空持仓文件"):
            empty = pd.DataFrame(columns=["ts_code", "buy_date", "shares", "weight"])
            empty.to_csv(positions_path, index=False, encoding="utf-8-sig")
            st.success(f"已创建空持仓文件：{positions_path}")


# -----------------------------
# 3. Run predict_daily.py
# -----------------------------
with tab_predict:
    st.header("3. 生成每日预测分数")
    st.write("调用 `src/predict_daily.py`，输出 `outputs_daily/latest_candidates.csv`。如果你已经有预测文件，也可以直接上传。")

    cand_upload = st.file_uploader("直接上传 latest_candidates.csv，可跳过预测脚本", type=["csv"], key="cand_upload")
    ensure_dir(outputs_daily)

    if cand_upload is not None:
        cand_df = pd.read_csv(cand_upload)
        cand_df.to_csv(latest_candidates_path, index=False, encoding="utf-8-sig")
        st.success(f"已保存预测文件：{latest_candidates_path}")

    st.subheader("调用预测脚本")
    cmd = [
        sys.executable,
        str(predict_script),
        "--data_dir",
        st.session_state.get("selected_data_dir", str(to_path(data_dir_manual))),
        "--checkpoint",
        str(to_path(checkpoint)),
        "--scaler",
        str(to_path(scaler)),
        "--positions",
        str(positions_path),
        "--out_dir",
        str(outputs_daily),
        "--seq_len",
        str(int(seq_len)),
        "--top_k_hold",
        str(int(topk)),
        "--drop_k",
        str(int(dropk)),
        "--trade_date",
        pd.to_datetime(trade_date).strftime("%Y-%m-%d"),
        "--device",
        device,
    ]
    if signal_date.strip():
        cmd.extend(["--signal_date", signal_date.strip()])

    st.code(" ".join(cmd), language="bash")

    if st.button("运行每日预测", type="primary", disabled=not parameter_ok):
        if not predict_script.exists():
            st.error(f"找不到预测脚本：{predict_script}")
        else:
            with st.spinner("正在运行 predict_daily.py ..."):
                code, output = run_command(cmd, cwd=project_root)
            st.text_area("运行日志", output, height=320)
            if code == 0:
                st.success("预测完成。")
            else:
                st.error(f"预测脚本运行失败，返回码：{code}")

    cand_df = read_csv_if_exists(latest_candidates_path)
    if not cand_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("预测股票数", len(cand_df))
        c2.metric("信号日期", infer_signal_date(cand_df) or "未知")
        if "pred" in cand_df.columns:
            c3.metric("最高预测分", f"{cand_df['pred'].max():.6f}")
        show_table("Top 50 最新预测", cand_df.head(50), height=420)
    else:
        st.info("尚未找到 latest_candidates.csv。")


# -----------------------------
# 4. Run make_trade_plan.py
# -----------------------------
with tab_plan:
    st.header("4. 生成比赛交易计划")
    st.write("调用 `src/make_trade_plan.py`，基于当前持仓和最新预测分数生成 buy/sell/hold。")

    ensure_dir(outputs_trade)

    cmd = [
        sys.executable,
        str(plan_script),
        "--pred",
        str(latest_candidates_path),
        "--positions",
        str(positions_path),
        "--out_dir",
        str(outputs_trade),
        "--trade_date",
        pd.to_datetime(trade_date).strftime("%Y-%m-%d"),
        "--topk",
        str(int(topk)),
        "--dropk",
        str(int(dropk)),
    ]
    if keep_missing_position:
        cmd.append("--keep_missing_position")

    st.code(" ".join(cmd), language="bash")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("生成交易计划", type="primary", disabled=not parameter_ok or signal_trade_ok is False):
            if not plan_script.exists():
                st.error(f"找不到交易计划脚本：{plan_script}")
            elif not latest_candidates_path.exists():
                st.error(f"找不到预测文件：{latest_candidates_path}")
            else:
                with st.spinner("正在运行 make_trade_plan.py ..."):
                    code, output = run_command(cmd, cwd=project_root)
                st.text_area("运行日志", output, height=320)
                if code == 0:
                    st.success("交易计划生成完成。")
                else:
                    st.error(f"交易计划脚本运行失败，返回码：{code}")

    with col2:
        confirm_clear = st.checkbox("确认清空交易计划输出目录", key="confirm_clear_trade_outputs")
        if st.button("清空交易计划输出目录", disabled=not confirm_clear):
            if outputs_trade.exists():
                shutil.rmtree(outputs_trade)
            ensure_dir(outputs_trade)
            st.success(f"已清空并重建：{outputs_trade}")

    plan_df = read_csv_if_exists(outputs_trade / "latest_trade_plan.csv")
    if not plan_df.empty:
        a, b, c, d = st.columns(4)
        a.metric("Sell", int((plan_df.get("action", pd.Series(dtype=str)) == "sell").sum()))
        b.metric("Buy", int((plan_df.get("action", pd.Series(dtype=str)) == "buy").sum()))
        c.metric("Hold", int((plan_df.get("action", pd.Series(dtype=str)) == "hold").sum()))
        d.metric("Total", len(plan_df))
        show_table("最新交易计划", plan_df, height=480)
    else:
        st.info("尚未找到 latest_trade_plan.csv。")


# -----------------------------
# 5. View and download
# -----------------------------
with tab_view:
    st.header("5. 查看与下载")

    buy_df = read_csv_if_exists(outputs_trade / "latest_buy_list.csv")
    sell_df = read_csv_if_exists(outputs_trade / "latest_sell_list.csv")
    hold_df = read_csv_if_exists(outputs_trade / "latest_hold_list.csv")
    plan_df = read_csv_if_exists(outputs_trade / "latest_trade_plan.csv")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        csv_download_button(buy_df, "下载 buy_list", "latest_buy_list.csv")
    with c2:
        csv_download_button(sell_df, "下载 sell_list", "latest_sell_list.csv")
    with c3:
        csv_download_button(hold_df, "下载 hold_list", "latest_hold_list.csv")
    with c4:
        csv_download_button(plan_df, "下载 trade_plan", "latest_trade_plan.csv")

    st.divider()
    t1, t2, t3, t4 = st.tabs(["Buy", "Sell", "Hold", "Full Plan"])
    with t1:
        show_table("买入清单", buy_df)
    with t2:
        show_table("卖出清单", sell_df)
    with t3:
        show_table("持有清单", hold_df)
    with t4:
        show_table("完整交易计划", plan_df, height=520)

    st.divider()
    st.subheader("明日同花顺操作提醒")
    st.markdown(
        """
1. 先处理 `sell_list`，释放现金。
2. 再按 `buy_list` 买入，尽量接近 `target_weight`。
3. `hold_list` 不动。
4. 成交后更新 `data/current_positions.csv`，第二天继续使用。
5. 保存同花顺持仓截图、成交截图和收益截图，方便最终报告使用。
        """.strip()
    )
