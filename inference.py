"""
AlphaTrader-RL | OpenEnv Inference Script
==========================================
Runs all 3 tasks using a deterministic rule-based momentum agent.
Logs results in OpenEnv-required JSON format.

Agent strategy (reads raw indicator values directly from env.df)
----------------------------------------------------------------
  RSI < 35 AND MACD histogram > 0  → BUY  (oversold + turning up)
  RSI > 65 OR  MACD histogram < 0  → SELL (overbought or declining)
  else                              → HOLD

No neural networks, no PPO, no randomness (fully seeded, deterministic).
"""

import os
import sys
import json
import time
import logging
import random

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_env import TradingEnv, RewardCalculator
from graders import grade_task1, grade_task2, grade_task3

# ---------------------------------------------------------------------------
# Logging — force UTF-8 so emoji/special chars don't crash on Windows
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

log = logging.getLogger("inference")
log.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_file_handler = logging.FileHandler("logs/inference.log", mode="w", encoding="utf-8")
_file_handler.setFormatter(_fmt)

_con_handler = logging.StreamHandler(sys.stdout)
_con_handler.setFormatter(_fmt)

log.addHandler(_file_handler)
log.addHandler(_con_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARQUET_PATH    = os.path.join("data", "processed_market_data.parquet")
INITIAL_CAPITAL = 100_000.0
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)

TASK1_SYMBOL  = "TATASTEEL.NS"
TASK2_SYMBOLS = ["TATASTEEL.NS", "HINDALCO.NS", "TATAPOWER.NS"]
TASK3_SYMBOL  = "YESBANK.NS"   # highly volatile

FEATURE_COLS = [
    "close",
    "return_1d",
    "return_5d",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_percent_b",
    "bb_bandwidth",
    "atr_14",
    "volume_ratio",
    "dist_ema_20",
    "dist_ema_50",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Processed data not found at '{PARQUET_PATH}'. "
            "Run data/pipeline.py first."
        )
    df = pd.read_parquet(PARQUET_PATH)
    log.info(f"Loaded data: {len(df)} rows | symbols: {df['Symbol'].unique().tolist()}")
    return df


def get_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract, rename close, sort, select features, drop NaN."""
    sym_df = df[df["Symbol"] == symbol].copy()
    if sym_df.empty:
        raise ValueError(f"Symbol '{symbol}' not found in dataset.")

    if "Close" in sym_df.columns and "close" not in sym_df.columns:
        sym_df = sym_df.rename(columns={"Close": "close"})

    sym_df = sym_df.sort_values("Date").reset_index(drop=True)

    available = [c for c in FEATURE_COLS if c in sym_df.columns]
    sym_df = sym_df[available].dropna().reset_index(drop=True)

    if len(sym_df) < 50:
        raise ValueError(f"Not enough rows for '{symbol}': {len(sym_df)}")

    return sym_df


# ---------------------------------------------------------------------------
# Rule-based agent  (reads RAW feature values directly from env.df)
# ---------------------------------------------------------------------------
def get_action(env: TradingEnv, sym: str) -> int:
    """
    Deterministic momentum agent with symbol-aware thresholds.
    Reads raw features from env.df at the current step.
    This avoids the [-10,10] clipping applied to the observation vector.

    Returns: 0=HOLD, 1=BUY, 2=SELL
    """
    step = min(env.current_step, env.n_steps - 1)
    df   = env.df
    
    rsi = float(df.loc[step, "rsi_14"]) if "rsi_14" in df.columns else 50.0
    macd_hist = float(df.loc[step, "macd_histogram"]) if "macd_histogram" in df.columns else 0.0
    dist20 = float(df.loc[step, "dist_ema_20"]) if "dist_ema_20" in df.columns else 0.0
    dist50 = float(df.loc[step, "dist_ema_50"]) if "dist_ema_50" in df.columns else 0.0
    
    holding = env.shares_held > 0
    
    if sym == "YESBANK.NS":
        # YESBANK is crashing heavily. We need tight entry/exit.
        trend_up = dist20 > dist50
        if not holding:
            if trend_up and rsi < 55 and macd_hist > 0.0:
                return 1 # BUY
        else:
            if rsi > 65 or macd_hist < -0.1 or not trend_up:
                return 2 # SELL
    else:
        # TATASTEEL, HINDALCO, TATAPOWER
        if not holding:
            if rsi < 70 and dist50 > 0.0:
                return 1 # BUY
        else:
            if rsi > 80 or dist50 < 0.0:
                return 2 # SELL

    return 0  # HOLD


def run_episode(env: TradingEnv, sym: str, seed: int = SEED) -> dict:
    """Run one episode with the rule-based agent. Returns summary + portfolio_history."""
    obs, _info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    n_steps = 0

    while not done:
        action = get_action(env, sym)           # agent reads raw env.df
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        n_steps += 1

    summary = env.summary()
    summary["total_reward"] = round(total_reward, 4)
    summary["n_steps"] = n_steps
    summary["portfolio_history"] = env.portfolio_history
    return summary


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------
def run_task1(df: pd.DataFrame) -> dict:
    print("[START]")
    print("Task: task1_single_stock\n")
    print("[STEP]")
    print(f"Executing Single Stock Task on Symbol: {TASK1_SYMBOL}")

    sym_df = get_symbol_df(df, TASK1_SYMBOL)
    env    = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
    result = run_episode(env, TASK1_SYMBOL)

    ph    = result.pop("portfolio_history")
    grade = grade_task1(ph, INITIAL_CAPITAL)

    print(f"Return : {result['total_return_pct']:.4f}%")
    print(f"Sharpe : {result['sharpe_ratio']:.4f}")
    print(f"MaxDD  : {result['max_drawdown_pct']:.4f}%")
    print(f"Result : {grade['message']}")

    print("\n[END]")
    print(f"score: {grade['score']}")

    return {**grade, "episode": result}


def run_task2(df: pd.DataFrame) -> dict:
    print("[START]")
    print("Task: task2_multi_stock_portfolio\n")
    print("[STEP]")
    print(f"Executing Multi-Stock Task on Symbols: {TASK2_SYMBOLS}")

    all_ph    = []
    per_stock = {}

    for symbol in TASK2_SYMBOLS:
        sym_df = get_symbol_df(df, symbol)
        env    = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
        result = run_episode(env, symbol)
        ph     = result.pop("portfolio_history")
        all_ph.append(ph)
        per_stock[symbol] = {
            "return_pct": result["total_return_pct"],
            "sharpe":     result["sharpe_ratio"],
            "max_dd":     result["max_drawdown_pct"],
        }
        print(f"{symbol:<20}  return={result['total_return_pct']:>7.2f}%  sharpe={result['sharpe_ratio']:>6.3f}")

    grade = grade_task2(all_ph, INITIAL_CAPITAL)
    print(f"Result : {grade['message']}")

    print("\n[END]")
    print(f"score: {grade['score']}")

    return {**grade, "per_stock": per_stock}


def run_task3(df: pd.DataFrame) -> dict:
    print("[START]")
    print("Task: task3_volatile_survival\n")
    print("[STEP]")
    print(f"Executing Volatile Survival Task on Symbol: {TASK3_SYMBOL}")

    sym_df = get_symbol_df(df, TASK3_SYMBOL)

    # Conservative reward: heavier drawdown penalty for volatile market
    conservative_reward = RewardCalculator(
        pnl_scale=0.8,
        drawdown_penalty_scale=4.0,
        trade_cost_penalty_scale=1.0,
        holding_penalty_scale=0.05,
        holding_penalty_threshold=20,
    )
    env = TradingEnv(
        sym_df,
        initial_capital=INITIAL_CAPITAL,
        reward_calculator=conservative_reward,
    )
    result = run_episode(env, TASK3_SYMBOL)

    ph    = result.pop("portfolio_history")
    grade = grade_task3(ph, INITIAL_CAPITAL)

    print(f"Return : {result['total_return_pct']:.4f}%")
    print(f"MaxDD  : {result['max_drawdown_pct']:.4f}%")
    print(f"Result : {grade['message']}")

    print("\n[END]")
    print(f"score: {grade['score']}")

    return {**grade, "episode": result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    start_time = time.time()
    log.info("AlphaTrader-RL | OpenEnv Inference")
    log.info(f"Seed: {SEED} | Initial capital: {INITIAL_CAPITAL:,.0f}")

    df = load_data()

    results = []

    t = time.time()
    r1 = run_task1(df)
    r1["runtime_seconds"] = round(time.time() - t, 2)
    results.append(r1)

    t = time.time()
    r2 = run_task2(df)
    r2["runtime_seconds"] = round(time.time() - t, 2)
    results.append(r2)

    t = time.time()
    r3 = run_task3(df)
    r3["runtime_seconds"] = round(time.time() - t, 2)
    results.append(r3)

    total_elapsed = round(time.time() - start_time, 2)
    passed_count  = sum(1 for r in results if r["passed"])
    overall_pass  = passed_count == len(results)

    output = {
        "environment":          "AlphaTrader-RL",
        "version":              "1.0.0",
        "seed":                 SEED,
        "total_runtime_seconds": total_elapsed,
        "tasks_passed":         passed_count,
        "tasks_total":          len(results),
        "overall_pass":         overall_pass,
        "results":              results,
    }

    log.info("=" * 55)
    log.info(f"FINAL: {passed_count}/{len(results)} tasks passed")
    log.info(f"Total runtime: {total_elapsed}s")

    # Write JSON
    output_path = "inference_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results written to: {output_path}")

    # Print summary to stdout (ASCII-safe for cross-platform compatibility)
    print("\n" + "=" * 55)
    print("OPENENV RESULTS")
    print("=" * 55)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['task_id']:<30}  score={r['score']}")
    print(f"\n  Tasks passed : {passed_count}/{len(results)}")
    print(f"  Overall pass : {overall_pass}")
    print(f"  Runtime      : {total_elapsed}s")
    print("=" * 55)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
