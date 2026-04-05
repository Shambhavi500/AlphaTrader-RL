"""
AlphaTrader-RL | OpenEnv Trading Environment
=============================================
Actions : 0=HOLD, 1=BUY, 2=SELL
Observation: market features + 5 portfolio-state scalars

OpenEnv API
-----------
  env.reset(seed)  -> (obs, info)
  env.step(action) -> (obs, reward, terminated, truncated, info)
  env.state()      -> dict with current portfolio metrics
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Reward calculator (self-contained, no external imports)
# ---------------------------------------------------------------------------
class RewardCalculator:
    """PnL-based reward with drawdown penalty and trade-cost penalty."""

    def __init__(
        self,
        pnl_scale: float = 1.0,
        drawdown_penalty_scale: float = 2.0,
        trade_cost_penalty_scale: float = 0.5,
        holding_penalty_scale: float = 0.1,
        holding_penalty_threshold: int = 10,
        reward_clip: float = 10.0,
    ):
        self.pnl_scale = pnl_scale
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.trade_cost_penalty_scale = trade_cost_penalty_scale
        self.holding_penalty_scale = holding_penalty_scale
        self.holding_penalty_threshold = holding_penalty_threshold
        self.reward_clip = reward_clip

    def calculate(
        self,
        prev_value: float,
        new_value: float,
        initial_capital: float,
        trade_executed: bool,
        trade_cost: float,
        steps_since_last_trade: int,
        position_size: int,
    ) -> float:
        step_return = (new_value - prev_value) / (initial_capital + 1e-8)

        r_pnl = step_return * self.pnl_scale * 100

        # Drawdown penalty on losses
        if new_value < prev_value:
            loss_pct = (prev_value - new_value) / (initial_capital + 1e-8)
            r_drawdown = -loss_pct * self.drawdown_penalty_scale * 100
        else:
            r_drawdown = 0.0

        # Transaction cost penalty
        if trade_executed and trade_cost > 0:
            cost_pct = trade_cost / (initial_capital + 1e-8)
            r_trade_cost = -cost_pct * self.trade_cost_penalty_scale * 100
        else:
            r_trade_cost = 0.0

        # Holding penalty (discourage stale positions)
        r_holding = 0.0
        if position_size > 0 and steps_since_last_trade > self.holding_penalty_threshold:
            excess = steps_since_last_trade - self.holding_penalty_threshold
            r_holding = -self.holding_penalty_scale * np.log1p(excess) * 0.01

        total = r_pnl + r_drawdown + r_trade_cost + r_holding
        return float(np.clip(total, -self.reward_clip, self.reward_clip))


# ---------------------------------------------------------------------------
# TradingEnv
# ---------------------------------------------------------------------------
class TradingEnv(gym.Env):
    """
    Single-stock gymnasium environment.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain a lower-case ``close`` column plus feature columns.
    initial_capital : float
    transaction_cost_pct : float
    max_position_size : float  (fraction of cash to invest)
    reward_calculator : RewardCalculator | None
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.0005,
        max_position_size: float = 0.95,
        reward_calculator: "RewardCalculator | None" = None,
        render_mode=None,
    ):
        super().__init__()

        if "close" not in features_df.columns:
            raise ValueError("features_df must contain a 'close' column (lower-case).")

        self.df = features_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_cols = [c for c in self.df.columns if c != "close"]
        self.n_features = len(self.feature_cols)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = max_position_size
        self.render_mode = render_mode
        self.reward_calc = reward_calculator or RewardCalculator()

        # Spaces
        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL
        obs_dim = self.n_features + 5            # features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_value = self._portfolio_value()
        price = self._current_price()
        trade_executed, trade_cost = False, 0.0

        if action == 1:   # BUY
            trade_executed, trade_cost = self._execute_buy(price)
        elif action == 2:  # SELL
            trade_executed, trade_cost = self._execute_sell(price)

        self.current_step += 1
        if trade_executed:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1

        new_value = self._portfolio_value()

        reward = self.reward_calc.calculate(
            prev_value=prev_value,
            new_value=new_value,
            initial_capital=self.initial_capital,
            trade_executed=trade_executed,
            trade_cost=trade_cost,
            steps_since_last_trade=self.steps_since_last_trade,
            position_size=self.shares_held,
        )

        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value

        self.portfolio_history.append(new_value)
        terminated = self.current_step >= self.n_steps - 1

        # Liquidate at episode end
        if terminated and self.shares_held > 0:
            self.cash += self.shares_held * self._current_price() * (
                1 - self.transaction_cost_pct
            )
            self.shares_held = 0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def state(self) -> dict:
        """
        OpenEnv state() — returns a snapshot of the current environment state.
        """
        pv = self._portfolio_value()
        ph = pd.Series(self.portfolio_history)
        returns = ph.pct_change().dropna()
        sharpe = float(
            returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        ) if len(returns) > 1 else 0.0
        max_dd = float(((ph - ph.cummax()) / ph.cummax()).min()) if len(ph) > 1 else 0.0

        return {
            "step": self.current_step,
            "total_steps": self.n_steps,
            "portfolio_value": round(pv, 2),
            "cash": round(self.cash, 2),
            "shares_held": self.shares_held,
            "total_return_pct": round(
                (pv - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "current_price": round(self._current_price(), 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares_held = 0
        self.peak_portfolio_value = self.initial_capital
        self.steps_since_last_trade = 0
        self.portfolio_history = [self.initial_capital]

    def _current_price(self) -> float:
        idx = min(self.current_step, self.n_steps - 1)
        return float(self.df.loc[idx, "close"])

    def _portfolio_value(self) -> float:
        return self.cash + self.shares_held * self._current_price()

    def _execute_buy(self, price: float):
        if self.shares_held > 0:
            return False, 0.0
        shares = int(
            (self.cash * self.max_position_size)
            / (price * (1 + self.transaction_cost_pct))
        )
        if shares <= 0:
            return False, 0.0
        cost = shares * price * (1 + self.transaction_cost_pct)
        self.cash -= cost
        self.shares_held += shares
        return True, cost * self.transaction_cost_pct

    def _execute_sell(self, price: float):
        if self.shares_held <= 0:
            return False, 0.0
        fee = self.shares_held * price * self.transaction_cost_pct
        self.cash += self.shares_held * price * (1 - self.transaction_cost_pct)
        self.shares_held = 0
        return True, fee

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, self.n_steps - 1)
        market_feats = self.df.loc[idx, self.feature_cols].values.astype(np.float32)
        pv = self._portfolio_value()
        price = self._current_price()
        portfolio_state = np.array(
            [
                self.cash / self.initial_capital - 1.0,
                (self.shares_held * price) / self.initial_capital,
                (pv - self.initial_capital) / self.initial_capital,
                (pv - self.peak_portfolio_value) / (self.peak_portfolio_value + 1e-8),
                min(self.steps_since_last_trade / 20.0, 1.0),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([market_feats, portfolio_state])
        return np.clip(np.nan_to_num(obs, nan=0.0), -10.0, 10.0)

    def _get_info(self) -> dict:
        pv = self._portfolio_value()
        return {
            "portfolio_value": pv,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "total_return_pct": (pv - self.initial_capital) / self.initial_capital * 100,
            "step": self.current_step,
            "price": self._current_price(),
        }

    def summary(self) -> dict:
        ph = pd.Series(self.portfolio_history)
        returns = ph.pct_change().dropna()
        return {
            "total_return_pct": round(
                (ph.iloc[-1] - self.initial_capital) / self.initial_capital * 100, 2
            ),
            "max_drawdown_pct": round(
                float(((ph - ph.cummax()) / ph.cummax()).min() * 100), 2
            ),
            "sharpe_ratio": round(
                float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)), 3
            ),
            "final_portfolio_value": round(float(ph.iloc[-1]), 2),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    parquet_path = os.path.join("data", "processed_market_data.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Run data/pipeline.py first. Expected: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # Pick one symbol, lower-case Close column
    symbol = "TATASTEEL.NS"
    sym_df = df[df["Symbol"] == symbol].copy()
    sym_df = sym_df.rename(columns={"Close": "close"})
    sym_df = sym_df.sort_values("Date").reset_index(drop=True)

    FEATURE_COLS = [
        "close", "return_1d", "return_5d", "rsi_14", "macd_line",
        "macd_signal", "macd_histogram", "bb_percent_b", "bb_bandwidth",
        "atr_14", "volume_ratio", "dist_ema_20", "dist_ema_50",
    ]
    available = [c for c in FEATURE_COLS if c in sym_df.columns]
    sym_df = sym_df[available].dropna().reset_index(drop=True)

    env = TradingEnv(sym_df, initial_capital=100_000)
    obs, info = env.reset(seed=42)
    print(f"Obs shape : {obs.shape}")
    print(f"State     : {env.state()}")

    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode done | Total reward: {total_reward:.4f}")
    print(f"Summary : {env.summary()}")
    print("✅ TradingEnv smoke test passed.")
