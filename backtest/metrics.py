"""
AlphaTrader-RL | Part 4: Backtesting Metrics
All metrics are computed from a portfolio-value time series and a trade log.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


class BacktestMetrics:
    """
    Computes standard backtesting performance metrics for an RL trading agent.

    Parameters
    ----------
    portfolio_history : list[float]
        Portfolio value at every step (length = n_steps + 1, first entry = initial capital).
    trade_log : list[dict]
        Each entry: {"step": int, "action": str, "price": float, "shares": int,
                     "cost": float, "portfolio_value": float}
    initial_capital : float
    prices : list[float]
        Raw close prices aligned with portfolio_history (used for Buy-and-Hold benchmark).
    """

    def __init__(
        self,
        portfolio_history: List[float],
        trade_log: List[dict],
        initial_capital: float,
        prices: List[float],
    ):
        self.portfolio = pd.Series(portfolio_history, dtype=float)
        self.trade_log = trade_log
        self.initial_capital = initial_capital
        self.prices = pd.Series(prices, dtype=float)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def total_return_pct(self) -> float:
        """Total portfolio return from start to end."""
        return (self.portfolio.iloc[-1] - self.initial_capital) / self.initial_capital * 100

    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown."""
        rolling_max = self.portfolio.cummax()
        drawdown = (self.portfolio - rolling_max) / rolling_max
        return float(drawdown.min() * 100)

    def sharpe_ratio(self, trading_days_per_year: int = 252) -> float:
        """Annualised Sharpe ratio (assumes risk-free rate ≈ 0)."""
        returns = self.portfolio.pct_change().dropna()
        if returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(trading_days_per_year))

    def sortino_ratio(self, trading_days_per_year: int = 252) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        returns = self.portfolio.pct_change().dropna()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() < 1e-10:
            return 0.0
        return float(returns.mean() / downside.std() * np.sqrt(trading_days_per_year))

    def calmar_ratio(self) -> float:
        """Annualised return divided by max drawdown (absolute)."""
        mdd = abs(self.max_drawdown_pct())
        if mdd < 1e-10:
            return 0.0
        n_years = len(self.portfolio) / 252
        ann_return = (self.total_return_pct() / 100) / max(n_years, 1e-10) * 100
        return float(ann_return / mdd)

    # ------------------------------------------------------------------
    # Trade stats
    # ------------------------------------------------------------------

    def num_trades(self) -> int:
        """Total number of executed trades (buys + sells)."""
        return len([t for t in self.trade_log if t.get("action") in ("BUY", "SELL")])

    def win_rate_pct(self) -> float:
        """
        Percentage of round-trips (buy → sell) that were profitable.
        A round-trip is profitable if the sell price > buy price.
        """
        buys = [t for t in self.trade_log if t.get("action") == "BUY"]
        sells = [t for t in self.trade_log if t.get("action") == "SELL"]
        pairs = list(zip(buys, sells))
        if not pairs:
            return 0.0
        wins = sum(1 for b, s in pairs if s["price"] > b["price"])
        return wins / len(pairs) * 100

    def avg_trade_pct(self) -> float:
        """Average PnL per round-trip as a percentage."""
        buys = [t for t in self.trade_log if t.get("action") == "BUY"]
        sells = [t for t in self.trade_log if t.get("action") == "SELL"]
        pairs = list(zip(buys, sells))
        if not pairs:
            return 0.0
        pnls = [(s["price"] - b["price"]) / b["price"] * 100 for b, s in pairs]
        return float(np.mean(pnls))

    # ------------------------------------------------------------------
    # Buy-and-Hold benchmark
    # ------------------------------------------------------------------

    def buy_and_hold_return_pct(self) -> float:
        """
        Return if the agent had simply bought at the first price and held
        until the last price (invested 95 % of capital, matching max_position_size).
        """
        if len(self.prices) < 2:
            return 0.0
        p0, p1 = float(self.prices.iloc[0]), float(self.prices.iloc[-1])
        shares = int(self.initial_capital * 0.95 / p0)
        if shares == 0:
            return 0.0
        final_value = self.initial_capital - shares * p0 + shares * p1
        return (final_value - self.initial_capital) / self.initial_capital * 100

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return all metrics as a single dict (values rounded for display)."""
        bah = self.buy_and_hold_return_pct()
        agent_ret = self.total_return_pct()
        return {
            "total_return_pct":         round(agent_ret, 2),
            "buy_hold_return_pct":      round(bah, 2),
            "alpha_pct":                round(agent_ret - bah, 2),
            "max_drawdown_pct":         round(self.max_drawdown_pct(), 2),
            "sharpe_ratio":             round(self.sharpe_ratio(), 3),
            "sortino_ratio":            round(self.sortino_ratio(), 3),
            "calmar_ratio":             round(self.calmar_ratio(), 3),
            "num_trades":               self.num_trades(),
            "win_rate_pct":             round(self.win_rate_pct(), 2),
            "avg_trade_pct":            round(self.avg_trade_pct(), 3),
            "final_portfolio_value":    round(float(self.portfolio.iloc[-1]), 2),
            "initial_capital":          round(self.initial_capital, 2),
        }
