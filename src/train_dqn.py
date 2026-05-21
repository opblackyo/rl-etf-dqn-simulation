"""Train a DQN agent on the historical 0050 ETF simulation environment."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.trading_env import ETFTradingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "0050_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_0050.zip"
REWARD_PATH = PROJECT_ROOT / "reports" / "training_rewards.csv"
SEED = 42


class RewardLoggerCallback(BaseCallback):
    """Collect per-step rewards during DQN training."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        reward = float(np.mean(self.locals.get("rewards", [0.0])))
        self.records.append({"step": int(self.num_timesteps), "reward": reward})
        return True


def load_train_data(feature_path: Path = FEATURE_PATH) -> pd.DataFrame:
    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特徵檔：{feature_path}，請先執行 python -m src.feature_engineering")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    if len(train_df) < 50:
        raise ValueError("訓練資料太少，請確認 data/0050_features.csv 是否完整。")
    return train_df


def train_dqn(total_timesteps: int = 20_000) -> DQN:
    random.seed(SEED)
    np.random.seed(SEED)
    set_random_seed(SEED)

    train_df = load_train_data()
    env = Monitor(ETFTradingEnv(train_df, random_seed=SEED))
    callback = RewardLoggerCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        seed=SEED,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    pd.DataFrame(callback.records).to_csv(REWARD_PATH, index=False, encoding="utf-8-sig")
    return model


def main() -> None:
    try:
        train_dqn()
    except Exception as exc:
        print(f"DQN 訓練失敗：{exc}")
        raise SystemExit(1) from exc

    print(f"模型已儲存到 {MODEL_PATH}")
    print(f"訓練 reward 已儲存到 {REWARD_PATH}")


if __name__ == "__main__":
    main()
