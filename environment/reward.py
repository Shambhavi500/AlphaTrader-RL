"""
AlphaTrader-RL | Part 2: Reward Function
"""
import numpy as np

class RewardCalculator:
    def __init__(self, pnl_scale=1.0, drawdown_penalty_scale=2.0,
                 trade_cost_penalty_scale=0.5, holding_penalty_scale=0.1,
                 holding_penalty_threshold=10, reward_clip=10.0):
        self.pnl_scale = pnl_scale
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.trade_cost_penalty_scale = trade_cost_penalty_scale
        self.holding_penalty_scale = holding_penalty_scale
        self.holding_penalty_threshold = holding_penalty_threshold
        self.reward_clip = reward_clip
        self._returns_window = []
        self._window_size = 50

    def calculate(self, prev_portfolio_value, new_portfolio_value, initial_capital,
                  trade_executed, trade_cost, trade_direction, steps_since_last_trade,
                  position_size, current_price):
        step_return = (new_portfolio_value - prev_portfolio_value) / (initial_capital + 1e-8)
        r_pnl = step_return * self.pnl_scale * 100
        self._returns_window.append(step_return)
        if len(self._returns_window) > self._window_size:
            self._returns_window.pop(0)
        if new_portfolio_value < prev_portfolio_value:
            loss_pct = (prev_portfolio_value - new_portfolio_value) / (initial_capital + 1e-8)
            r_drawdown = -loss_pct * self.drawdown_penalty_scale * 100
        else:
            r_drawdown = 0.0
        if trade_executed and trade_cost > 0:
            cost_pct = trade_cost / (initial_capital + 1e-8)
            r_trade_cost = -cost_pct * self.trade_cost_penalty_scale * 100
        else:
            r_trade_cost = 0.0
        r_holding = 0.0
        if position_size > 0 and steps_since_last_trade > self.holding_penalty_threshold:
            excess_steps = steps_since_last_trade - self.holding_penalty_threshold
            r_holding = -self.holding_penalty_scale * np.log1p(excess_steps) * 0.01
        r_sharpe_bonus = 0.0
        if len(self._returns_window) >= 10:
            mean_ret = np.mean(self._returns_window)
            std_ret = np.std(self._returns_window) + 1e-8
            rolling_sharpe = mean_ret / std_ret
            r_sharpe_bonus = np.clip(rolling_sharpe * 0.005, -0.5, 0.5)
        total_reward = r_pnl + r_drawdown + r_trade_cost + r_holding + r_sharpe_bonus
        return float(np.clip(total_reward, -self.reward_clip, self.reward_clip))

    def reset(self):
        self._returns_window = []

    def get_config(self):
        return {
            "pnl_scale": self.pnl_scale,
            "drawdown_penalty_scale": self.drawdown_penalty_scale,
            "trade_cost_penalty_scale": self.trade_cost_penalty_scale,
            "holding_penalty_scale": self.holding_penalty_scale,
            "holding_penalty_threshold": self.holding_penalty_threshold,
            "reward_clip": self.reward_clip,
        }

class ConservativeReward(RewardCalculator):
    def __init__(self):
        super().__init__(pnl_scale=0.8, drawdown_penalty_scale=4.0,
                         trade_cost_penalty_scale=1.0, holding_penalty_scale=0.05,
                         holding_penalty_threshold=20)

class AggressiveReward(RewardCalculator):
    def __init__(self):
        super().__init__(pnl_scale=2.0, drawdown_penalty_scale=0.5,
                         trade_cost_penalty_scale=0.2, holding_penalty_scale=0.2,
                         holding_penalty_threshold=5)
