"""Performance metrics for historical trading simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """Calculate win rate from paired Buy/Sell trades."""
    if trades.empty or "Action" not in trades.columns:
        return np.nan

    completed_returns: list[float] = []
    entry_price: float | None = None

    for _, trade in trades.iterrows():
        if trade["Action"] == "Buy":
            entry_price = float(trade["Price"])
        elif trade["Action"] == "Sell" and entry_price is not None:
            exit_price = float(trade["Price"])
            completed_returns.append((exit_price - entry_price) / entry_price)
            entry_price = None

    if not completed_returns:
        return np.nan
    wins = sum(1 for value in completed_returns if value > 0)
    return float(wins / len(completed_returns))


def performance_summary(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    strategy_name: str = "Strategy",
) -> dict[str, float | str]:
    """Return a compact performance summary for one strategy."""
    if equity_curve.empty or "Equity" not in equity_curve.columns:
        raise ValueError("equity_curve must contain an Equity column.")

    trades = pd.DataFrame() if trades is None else trades
    equity = equity_curve["Equity"].astype(float)
    initial_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])
    total_return = final_equity / initial_equity - 1.0 if initial_equity else 0.0

    return {
        "strategy": strategy_name,
        "total_return": float(total_return),
        "max_drawdown": calculate_max_drawdown(equity),
        "num_trades": int(len(trades)),
        "win_rate": calculate_win_rate(trades),
        "final_equity": final_equity,
    }
