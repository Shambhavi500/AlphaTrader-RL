"""
AlphaTrader-RL | Part 2: Trading Environment
Actions: 0=HOLD, 1=BUY, 2=SELL
Observation: 45 market features + 5 portfolio state = 50-dim vector
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from environment.reward import RewardCalculator

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, features_df, initial_capital=100_000.0,
                 transaction_cost_pct=0.0005, max_position_size=0.95,
                 render_mode=None):
        super().__init__()
        if "close" not in features_df.columns:
            raise ValueError("features_df must contain a 'close' column.")
        self.df = features_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_cols = [c for c in self.df.columns if c != "close"]
        self.n_features = len(self.feature_cols)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = max_position_size
        self.render_mode = render_mode
        self.reward_calc = RewardCalculator()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.n_features + 5,), dtype=np.float32)
        self._reset_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        prev_value = self._portfolio_value()
        price = self._current_price()
        trade_executed, trade_cost, trade_direction = False, 0.0, 0
        if action == 1:
            trade_executed, trade_cost = self._execute_buy(price)
            if trade_executed: trade_direction = 1
        elif action == 2:
            trade_executed, trade_cost = self._execute_sell(price)
            if trade_executed: trade_direction = -1
        self.current_step += 1
        self.steps_since_last_trade = 0 if trade_executed else self.steps_since_last_trade + 1
        new_value = self._portfolio_value()
        reward = self.reward_calc.calculate(
            prev_portfolio_value=prev_value, new_portfolio_value=new_value,
            initial_capital=self.initial_capital, trade_executed=trade_executed,
            trade_cost=trade_cost, trade_direction=trade_direction,
            steps_since_last_trade=self.steps_since_last_trade,
            position_size=self.shares_held, current_price=self._current_price())
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
        self.portfolio_history.append(new_value)
        terminated = self.current_step >= self.n_steps - 1
        if terminated and self.shares_held > 0:
            self.cash += self.shares_held * self._current_price() * (1 - self.transaction_cost_pct)
            self.shares_held = 0
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reset_state(self):
        self.current_step = 0
        self.cash = float(self.initial_capital)
        self.shares_held = 0
        self.peak_portfolio_value = float(self.initial_capital)
        self.steps_since_last_trade = 0
        self.portfolio_history = [float(self.initial_capital)]

    def _current_price(self):
        return float(self.df.loc[min(self.current_step, self.n_steps - 1), "close"])

    def _portfolio_value(self):
        return self.cash + self.shares_held * self._current_price()

    def _execute_buy(self, price):
        if self.shares_held > 0: return False, 0.0
        shares = int((self.cash * self.max_position_size) / (price * (1 + self.transaction_cost_pct)))
        if shares <= 0: return False, 0.0
        cost = shares * price * (1 + self.transaction_cost_pct)
        self.cash -= cost
        self.shares_held += shares
        return True, cost * self.transaction_cost_pct

    def _execute_sell(self, price):
        if self.shares_held <= 0: return False, 0.0
        fee = self.shares_held * price * self.transaction_cost_pct
        self.cash += self.shares_held * price * (1 - self.transaction_cost_pct)
        self.shares_held = 0
        return True, fee

    def _get_observation(self):
        idx = min(self.current_step, self.n_steps - 1)
        market_features = self.df.loc[idx, self.feature_cols].values.astype(np.float32)
        pv = self._portfolio_value()
        price = self._current_price()
        portfolio_state = np.array([
            self.cash / self.initial_capital - 1.0,
            (self.shares_held * price) / self.initial_capital,
            (pv - self.initial_capital) / self.initial_capital,
            (pv - self.peak_portfolio_value) / (self.peak_portfolio_value + 1e-8),
            min(self.steps_since_last_trade / 20.0, 1.0)
        ], dtype=np.float32)
        obs = np.concatenate([market_features, portfolio_state])
        return np.clip(np.nan_to_num(obs, nan=0.0), -10.0, 10.0)

    def _get_info(self):
        pv = self._portfolio_value()
        return {
            "portfolio_value": pv,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "total_return_pct": (pv - self.initial_capital) / self.initial_capital * 100,
            "current_step": self.current_step,
            "current_price": self._current_price(),
        }

    def summary(self):
        pv = pd.Series(self.portfolio_history)
        returns = pv.pct_change().dropna()
        return {
            "total_return_pct": round((pv.iloc[-1] - self.initial_capital) / self.initial_capital * 100, 2),
            "max_drawdown_pct": round(((pv - pv.cummax()) / pv.cummax()).min() * 100, 2),
            "sharpe_ratio": round(float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)), 3),
            "final_portfolio_value": round(float(pv.iloc[-1]), 2),
        }
