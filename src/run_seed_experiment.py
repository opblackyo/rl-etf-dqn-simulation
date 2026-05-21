"""Run DQN backtests across multiple random seeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3.common.monitor import Monitor

from src.backtest import INITIAL_CASH, TRANSACTION_COST
from src.metrics import performance_summary
from src.trading_env import ETFTradingEnv
from src.train_dqn import (
    FEATURE_PATH,
    TOTAL_TIMESTEPS,
    RewardLoggerCallback,
    create_dqn_model,
    set_all_seeds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORT_DIR / "seed_experiment_summary.csv"
SEEDS = [42, 123, 999]


def load_train_test_data(feature_path: Path = FEATURE_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特徵檔：{feature_path}，請先執行 python -m src.feature_engineering")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    if len(train_df) < 50 or len(test_df) < 20:
        raise ValueError("資料筆數不足，請確認 data/0050_features.csv 是否完整。")
    return train_df, test_df


def run_policy_on_test(model, test_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = ETFTradingEnv(
        test_df,
        initial_cash=INITIAL_CASH,
        transaction_cost=TRANSACTION_COST,
        random_seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
    return env.get_equity_curve(), env.get_trades()


def run_seed_experiment() -> pd.DataFrame:
    train_df, test_df = load_train_test_data()
    rows = []

    for seed in SEEDS:
        print(f"開始 seed={seed} DQN 訓練與回測")
        set_all_seeds(seed)
        env = Monitor(
            ETFTradingEnv(
                train_df,
                initial_cash=INITIAL_CASH,
                transaction_cost=TRANSACTION_COST,
                random_seed=seed,
            )
        )
        callback = RewardLoggerCallback()
        model = create_dqn_model(env, seed)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        equity_curve, trades = run_policy_on_test(model, test_df, seed)
        row = performance_summary(equity_curve, trades, f"seed_{seed}")
        row["seed"] = seed
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary[["seed", "final_equity", "total_return", "max_drawdown", "num_trades", "win_rate"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    try:
        summary = run_seed_experiment()
    except Exception as exc:
        print(f"多 seed 實驗失敗：{exc}")
        raise SystemExit(1) from exc

    print(f"多 seed 實驗完成，結果已儲存到 {OUTPUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
