"""
AlphaTrader-RL | Part 6: Paper Trader
Simulates live trading by executing decisions step-by-step and explaining them via LLM.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from agent.ppo_agent import PPOAgent
from environment.trading_env import TradingEnv
from llm.explainer import explain_trade

logger = logging.getLogger("PaperTrader")

class PaperTrader:
    """
    Handles live/simulated trading logic for a single symbol.
    """
    def __init__(self, model_path: str, symbol: str, initial_capital: float = 100_000.0):
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Load a dummy environment to initialize the agent (shape-only usually)
        # In a real live scenario, we'd need a way to construct the observation vector from live data.
        # For this simulation, we'll assume observations are provided.
        self.agent = None
        self.model_path = model_path
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.shares_held = 0
        self.trade_history = []
        self._is_loaded = False

    def load_agent(self, env: TradingEnv):
        """Standard loading using the environment for shape/space info."""
        if not os.path.exists(self.model_path + ".zip"):
            raise FileNotFoundError(f"Model {self.model_path}.zip not found.")
        self.agent = PPOAgent.load(self.model_path, env=env)
        self._is_loaded = True
        logger.info(f"Agent loaded for {self.symbol}")

    def execute_step(self, observation: np.ndarray, price: float, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a single market observation, makes a decision, and returns results + explanation.
        """
        if not self._is_loaded:
            return {"error": "Agent not loaded"}

        # 1. Predict action (0=HOLD, 1=BUY, 2=SELL)
        action_idx, _ = self.agent.model.predict(observation, deterministic=True)
        action_idx = int(action_idx)
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[action_idx]

        # 2. Update local portfolio state (simple simulation)
        # Note: In a production system, this would interact with a broker API.
        trade_occurred = False
        if action_idx == 1 and self.shares_held == 0:  # BUY
            shares_to_buy = int((self.cash * 0.95) / price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares_held = shares_to_buy
                trade_occurred = True
        elif action_idx == 2 and self.shares_held > 0:  # SELL
            self.cash += self.shares_held * price
            self.shares_held = 0
            trade_occurred = True

        self.portfolio_value = self.cash + (self.shares_held * price)
        total_return_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital * 100

        # 3. Get LLM explanation
        portfolio_info = {
            "total_return_pct": total_return_pct,
            "shares_held": self.shares_held,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value
        }
        
        try:
            explanation = explain_trade(action_name, feature_dict, portfolio_info)
        except Exception as e:
            logger.error(f"LLM Explanation failed: {e}")
            explanation = f"[AI Explanation Error] {e}"

        result = {
            "symbol": self.symbol,
            "action": action_name,
            "price": price,
            "portfolio_value": self.portfolio_value,
            "total_return_pct": total_return_pct,
            "shares_held": self.shares_held,
            "explanation": explanation,
            "trade_occurred": trade_occurred
        }
        
        if trade_occurred:
            self.trade_history.append(result)
            
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "total_trades": len(self.trade_history)
        }
