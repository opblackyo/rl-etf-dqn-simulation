"""Gymnasium environment for simplified ETF trading simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ETFTradingEnv(gym.Env):
    """A simple all-in/all-out trading environment for 0050 ETF.

    This environment is for historical simulation and education only. It has no
    broker API integration and cannot place real orders.
    """

    metadata = {"render_modes": []}

    feature_columns = [
        "return_1d",
        "ma_ratio",
        "volatility_20",
        "volume_change",
        "position",
        "cash_ratio",
    ]
    trade_columns = ["Date", "Action", "Price", "Shares", "Cash", "NetWorth"]

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001425,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        if data.empty:
            raise ValueError("ETFTradingEnv requires non-empty data.")

        self.data = data.sort_values("Date").reset_index(drop=True).copy()
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(random_seed)
        self.reset(seed=random_seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0
        self.net_worth = self.initial_cash
        self.prev_net_worth = self.initial_cash
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = [self._equity_record(action=0, reward=0.0)]
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0=Hold, 1=Buy, or 2=Sell.")

        price = self._current_price()
        self.prev_net_worth = self._calculate_net_worth(price)
        executed_action = "Hold"

        if action == 1 and self.position == 0:
            cost_multiplier = 1.0 + self.transaction_cost
            self.shares = self.cash / (price * cost_multiplier)
            self.cash = 0.0
            self.position = 1
            executed_action = "Buy"
            self._record_trade(executed_action, price)
        elif action == 2 and self.position == 1:
            self.cash = self.shares * price * (1.0 - self.transaction_cost)
            self.shares = 0.0
            self.position = 0
            executed_action = "Sell"
            self._record_trade(executed_action, price)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        new_price = self._current_price()
        self.net_worth = self._calculate_net_worth(new_price)
        reward = (self.net_worth - self.prev_net_worth) / max(self.prev_net_worth, 1e-8)
        self.equity_curve.append(self._equity_record(action=action, reward=reward))

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def get_trades(self) -> pd.DataFrame:
        """Return the complete trade log generated in the current episode."""
        return pd.DataFrame(self.trades, columns=self.trade_columns)

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve generated in the current episode."""
        return pd.DataFrame(self.equity_curve)

    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]
        cash_ratio = self.cash / max(self.net_worth, 1e-8)
        obs = np.array(
            [
                row["return_1d"],
                row["ma_ratio"],
                row["volatility_20"],
                row["volume_change"],
                float(self.position),
                cash_ratio,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_info(self) -> dict[str, Any]:
        return {
            "date": self.data.iloc[self.current_step]["Date"],
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "net_worth": self.net_worth,
            "trades": self.trades,
        }

    def _current_price(self) -> float:
        return float(self.data.iloc[self.current_step]["Close"])

    def _calculate_net_worth(self, price: float) -> float:
        return float(self.cash + self.shares * price)

    def _record_trade(self, action_name: str, price: float) -> None:
        row = self.data.iloc[self.current_step]
        self.trades.append(
            {
                "Date": row["Date"],
                "Action": action_name,
                "Price": price,
                "Shares": self.shares,
                "Cash": self.cash,
                "NetWorth": self._calculate_net_worth(price),
            }
        )

    def _equity_record(self, action: int, reward: float) -> dict[str, Any]:
        row = self.data.iloc[self.current_step]
        price = self._current_price()
        return {
            "Date": row["Date"],
            "Close": price,
            "Action": int(action),
            "Position": int(self.position),
            "Cash": self.cash,
            "Shares": self.shares,
            "Equity": self._calculate_net_worth(price),
            "Reward": float(reward),
        }
