"""Backtest DQN, buy-and-hold, and random policies on held-out 0050 data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from src.metrics import performance_summary
from src.trading_env import ETFTradingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "0050_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_0050.zip"
REPORT_DIR = PROJECT_ROOT / "reports"
INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001425
SEED = 42


def load_test_data(feature_path: Path = FEATURE_PATH) -> pd.DataFrame:
    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特徵檔：{feature_path}，請先執行 python -m src.feature_engineering")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    if len(test_df) < 20:
        raise ValueError("測試資料太少，請確認 data/0050_features.csv 是否完整。")
    return test_df


def run_dqn_policy(test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔：{MODEL_PATH}，請先執行 python -m src.train_dqn")

    env = ETFTradingEnv(test_df, initial_cash=INITIAL_CASH, transaction_cost=TRANSACTION_COST, random_seed=SEED)
    model = DQN.load(MODEL_PATH, env=env)

    obs, _ = env.reset(seed=SEED)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))

    return env.get_equity_curve(), env.get_trades()


def run_buy_and_hold(test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = test_df.sort_values("Date").reset_index(drop=True)
    first_price = float(df.iloc[0]["Close"])
    shares = INITIAL_CASH / (first_price * (1.0 + TRANSACTION_COST))
    trades = pd.DataFrame(
        [
            {
                "Date": df.iloc[0]["Date"],
                "Action": "Buy",
                "Price": first_price,
                "Shares": shares,
                "Cash": 0.0,
                "NetWorth": shares * first_price,
            }
        ]
    )
    equity = df[["Date", "Close"]].copy()
    equity["Action"] = 0
    equity.loc[0, "Action"] = 1
    equity["Position"] = 1
    equity["Cash"] = 0.0
    equity["Shares"] = shares
    equity["Equity"] = shares * equity["Close"].astype(float)
    equity["Reward"] = equity["Equity"].pct_change().fillna(0.0)
    return equity, trades


def run_random_policy(test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    env = ETFTradingEnv(test_df, initial_cash=INITIAL_CASH, transaction_cost=TRANSACTION_COST, random_seed=SEED)
    _, _ = env.reset(seed=SEED)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(0, 3))
        _, _, terminated, truncated, _ = env.step(action)
    return env.get_equity_curve(), env.get_trades()


def run_backtest() -> pd.DataFrame:
    test_df = load_test_data()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dqn_equity, dqn_trades = run_dqn_policy(test_df)
    buy_hold_equity, buy_hold_trades = run_buy_and_hold(test_df)
    random_equity, random_trades = run_random_policy(test_df)

    dqn_equity.to_csv(REPORT_DIR / "dqn_equity_curve.csv", index=False, encoding="utf-8-sig")
    buy_hold_equity.to_csv(REPORT_DIR / "buy_hold_equity_curve.csv", index=False, encoding="utf-8-sig")
    random_equity.to_csv(REPORT_DIR / "random_equity_curve.csv", index=False, encoding="utf-8-sig")
    dqn_trades.to_csv(REPORT_DIR / "trades.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            performance_summary(dqn_equity, dqn_trades, "DQN"),
            performance_summary(buy_hold_equity, buy_hold_trades, "Buy and Hold"),
            performance_summary(random_equity, random_trades, "Random Policy"),
        ]
    )
    summary.to_csv(REPORT_DIR / "performance_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    try:
        summary = run_backtest()
    except Exception as exc:
        print(f"回測失敗：{exc}")
        raise SystemExit(1) from exc

    print("回測完成，績效摘要：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
