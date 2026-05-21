"""Streamlit dashboard for the 0050 DQN trading simulation project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_0050.zip"

DAILY_PATH = DATA_DIR / "0050_daily.csv"
FEATURE_PATH = DATA_DIR / "0050_features.csv"
REWARD_PATH = REPORT_DIR / "training_rewards.csv"
DQN_EQUITY_PATH = REPORT_DIR / "dqn_equity_curve.csv"
BUY_HOLD_EQUITY_PATH = REPORT_DIR / "buy_hold_equity_curve.csv"
RANDOM_EQUITY_PATH = REPORT_DIR / "random_equity_curve.csv"
TRADES_PATH = REPORT_DIR / "trades.csv"
SUMMARY_PATH = REPORT_DIR / "performance_summary.csv"
SEED_EXPERIMENT_PATH = REPORT_DIR / "seed_experiment_summary.csv"
REWARD_EXPERIMENT_PATH = REPORT_DIR / "reward_experiment_summary.csv"


st.set_page_config(
    page_title="0050 DQN 交易決策模擬",
    layout="wide",
)


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


def show_missing_file_warning() -> bool:
    missing_steps = []
    if not DAILY_PATH.exists():
        missing_steps.append("尚未找到 0050 日資料，請先執行 `python -m src.data_loader`")
    if not FEATURE_PATH.exists():
        missing_steps.append("尚未找到特徵資料，請先執行 `python -m src.feature_engineering`")
    if not MODEL_PATH.exists():
        missing_steps.append("尚未找到模型檔，請先執行 `python -m src.train_dqn`")
    if not DQN_EQUITY_PATH.exists() or not SUMMARY_PATH.exists():
        missing_steps.append("尚未找到回測報表，請先執行 `python -m src.backtest`")

    if missing_steps:
        st.warning("部分資料尚未產生。")
        for step in missing_steps:
            st.write(step)
        return True
    return False


def get_split_info(feature_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    split_index = int(len(feature_df) * 0.8)
    train_df = feature_df.iloc[:split_index]
    test_df = feature_df.iloc[split_index:]
    return {
        "train_start": train_df["Date"].min(),
        "train_end": train_df["Date"].max(),
        "test_start": test_df["Date"].min(),
        "test_end": test_df["Date"].max(),
        "split_date": test_df["Date"].min(),
    }


def plot_price(daily_df: pd.DataFrame, split_date: pd.Timestamp | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Close"],
            mode="lines",
            name="0050 Close",
            line=dict(color="#2563eb", width=2),
        )
    )
    if split_date is not None:
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Test start",
            annotation_position="top",
        )
    fig.update_layout(
        title="0050 價格走勢",
        xaxis_title="Date",
        yaxis_title="Close",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def show_split_info(feature_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    split_info = get_split_info(feature_df)
    st.subheader("資料切分說明")
    st.write("本專題採用時間序列切分，前 80% 作為 DQN 訓練資料，後 20% 作為測試與回測資料。")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Train period",
            f"{split_info['train_start'].date()} ~ {split_info['train_end'].date()}",
        )
    with col2:
        st.metric(
            "Test period",
            f"{split_info['test_start'].date()} ~ {split_info['test_end'].date()}",
        )
    return split_info


def plot_equity_curves(dqn_df: pd.DataFrame, buy_hold_df: pd.DataFrame, random_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dqn_df["Date"], y=dqn_df["Equity"], mode="lines", name="DQN"))
    fig.add_trace(
        go.Scatter(x=buy_hold_df["Date"], y=buy_hold_df["Equity"], mode="lines", name="Buy and Hold")
    )
    fig.add_trace(
        go.Scatter(x=random_df["Date"], y=random_df["Equity"], mode="lines", name="Random Policy")
    )
    fig.update_layout(
        title="策略淨值曲線比較",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_dqn_signals(dqn_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dqn_df["Date"],
            y=dqn_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#334155", width=2),
        )
    )

    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["Date"] = pd.to_datetime(trades_df["Date"])
        buys = trades_df[trades_df["Action"] == "Buy"]
        sells = trades_df[trades_df["Action"] == "Sell"]
        fig.add_trace(
            go.Scatter(
                x=buys["Date"],
                y=buys["Price"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", color="#16a34a", size=12),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sells["Date"],
                y=sells["Price"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", color="#dc2626", size=12),
            )
        )

    fig.update_layout(
        title="DQN 交易訊號",
        xaxis_title="Date",
        yaxis_title="Close",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_rewards(reward_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reward_df["step"],
            y=reward_df["reward"],
            mode="lines",
            name="Training Reward",
            line=dict(color="#7c3aed", width=1.5),
        )
    )
    fig.update_layout(
        title="DQN 訓練 Reward 曲線",
        xaxis_title="Training Step",
        yaxis_title="Reward",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def format_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    formatted = summary_df.copy()
    for column in ["total_return", "max_drawdown", "win_rate"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].apply(
                lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
            )
    if "final_equity" in formatted.columns:
        formatted["final_equity"] = formatted["final_equity"].apply(lambda value: f"{float(value):,.0f}")
    return formatted


st.title("強化學習於台股 ETF 交易決策模擬與視覺化系統")
st.info("本專案僅作為強化學習與金融資料分析的自發學習成果，不涉及真實下單，也不構成任何投資建議。")

has_missing_files = show_missing_file_warning()
split_info = None

if FEATURE_PATH.exists():
    features = read_csv(FEATURE_PATH)
    split_info = show_split_info(features)

if DAILY_PATH.exists():
    daily = read_csv(DAILY_PATH)
    split_date = split_info["split_date"] if split_info is not None else None
    st.plotly_chart(plot_price(daily, split_date), use_container_width=True)

if not has_missing_files:
    dqn_equity = read_csv(DQN_EQUITY_PATH)
    buy_hold_equity = read_csv(BUY_HOLD_EQUITY_PATH)
    random_equity = read_csv(RANDOM_EQUITY_PATH)
    trades = read_csv(TRADES_PATH) if TRADES_PATH.exists() and TRADES_PATH.stat().st_size > 0 else pd.DataFrame()
    rewards = pd.read_csv(REWARD_PATH) if REWARD_PATH.exists() else pd.DataFrame()
    summary = pd.read_csv(SUMMARY_PATH)

    st.plotly_chart(plot_equity_curves(dqn_equity, buy_hold_equity, random_equity), use_container_width=True)
    st.plotly_chart(plot_dqn_signals(dqn_equity, trades), use_container_width=True)

    if not rewards.empty:
        st.plotly_chart(plot_rewards(rewards), use_container_width=True)
    else:
        st.warning("尚未找到 reward 紀錄，請確認 `reports/training_rewards.csv` 是否存在。")

    st.subheader("績效表格")
    st.dataframe(format_summary(summary), use_container_width=True, hide_index=True)

if SEED_EXPERIMENT_PATH.exists():
    st.subheader("多 Seed 實驗")
    st.write("多 seed 實驗用來觀察 DQN 結果穩定性；不同 random seed 可能造成訓練路徑與交易結果不同。")
    seed_summary = pd.read_csv(SEED_EXPERIMENT_PATH)
    st.dataframe(format_summary(seed_summary), use_container_width=True, hide_index=True)

if REWARD_EXPERIMENT_PATH.exists():
    st.subheader("Reward Function 比較")
    st.write("reward 比較用來觀察 reward function 對交易行為的影響，結果只代表歷史資料模擬，不代表投資建議。")
    reward_summary = pd.read_csv(REWARD_EXPERIMENT_PATH)
    st.dataframe(format_summary(reward_summary), use_container_width=True, hide_index=True)
