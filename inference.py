"""
AlphaTrader-RL | OpenEnv Inference Script
==========================================
Runs all 3 tasks using a deterministic rule-based momentum agent.
Logs results in OpenEnv-required JSON format.

Agent strategy — task-specific logic
--------------------------------------
  TASK 1 (Easy — single stock profit):
    BUY  if RSI < 70 AND dist_ema50 > -0.01 AND macd_hist > -0.05
    SELL if RSI > 75 OR dist_ema50 < -0.02

  TASK 2 (Medium — multi-stock Sharpe):
    BUY  if RSI < 55 AND macd_hist > 0 AND dist_ema50 > 0   (strict — only clear uptrends)
    SELL if RSI > 65 OR dist_ema50 < -0.015                 (exit faster to protect Sharpe)

  TASK 3 (Hard — volatile survival):
    BUY  if RSI < 45 AND macd_hist > 0 AND dist_ema50 > 0   (only strong confirmed entries)
    SELL if RSI > 70 OR dist_ema50 < -0.01                  (exit early)
         OR live drawdown from peak > 8%                    (hard stop-loss)
"""

import os
import sys
import json
import time
import logging
import random
import argparse
import yfinance as yf

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import FeatureEngineer

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
TASK2_SYMBOLS = ["TATASTEEL.NS", "GOLDBEES.NS", "SILVERBEES.NS"]  # was HINDALCO, TATAPOWER
TASK3_SYMBOL  = "YESBANK.NS"

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
        print(f"Symbol '{symbol}' missing from local cache. Auto-fetching via yfinance...")
        raw_df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True)
        if raw_df.empty:
            raise ValueError(f"yfinance failed to fetch data for '{symbol}'.")
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        
        raw_df = raw_df.reset_index()
        if "index" in raw_df.columns:
            raw_df = raw_df.rename(columns={"index": "Date"})
        raw_df['Symbol'] = symbol

        try:
            eng = FeatureEngineer()
            eng_df = eng.add_price_features(raw_df)
            eng_df = eng.add_technical_indicators(eng_df)
            eng_df = eng.add_volume_features(eng_df)
        except Exception as e:
            raise RuntimeError(f"Feature engineering failed for '{symbol}': {e}")
        
        eng_df = eng_df.replace([np.inf, -np.inf], np.nan).dropna()

        if os.path.exists(PARQUET_PATH):
            existing_df = pd.read_parquet(PARQUET_PATH)
            for col in existing_df.columns:
                if col not in eng_df.columns:
                    eng_df[col] = np.nan
            eng_df = eng_df[existing_df.columns]
            
            updated_df = pd.concat([existing_df, eng_df], ignore_index=True)
            updated_df.to_parquet(PARQUET_PATH)
            log.info(f"Updated global parquet with {symbol}.")
            
        sym_df = eng_df.copy()

    if "Close" in sym_df.columns and "close" not in sym_df.columns:
        sym_df = sym_df.rename(columns={"Close": "close"})

    sym_df = sym_df.sort_values("Date").reset_index(drop=True)

    available = [c for c in FEATURE_COLS if c in sym_df.columns]
    
    print(f"\n--- Validation for {symbol} ---")
    print(sym_df[available].head())
    
    critical = ["rsi_14", "macd_histogram", "dist_ema_50", "close"]
    for c in critical:
        if c in sym_df.columns:
            nan_count = sym_df[c].isna().sum()
            if nan_count > (len(sym_df) * 0.5):
                 log.warning(f"Critical column '{c}' for '{symbol}' has {nan_count} NaNs.")
    
    sym_df = sym_df[available].dropna().reset_index(drop=True)
    print(f"Final usable rows for {symbol}: {len(sym_df)}")
    
    if len(sym_df) < 50:
        raise ValueError(f"Not enough usable rows for '{symbol}': {len(sym_df)}")

    return sym_df


# ---------------------------------------------------------------------------
# Rule-based agent  — task-specific logic
# ---------------------------------------------------------------------------
def get_action(env: TradingEnv, sym: str, task_id: str = "task1") -> int:
    """
    Task-specific deterministic momentum agent.

    task_id: "task1" | "task2" | "task3"
    Returns: 0=HOLD, 1=BUY, 2=SELL
    """
    step = min(env.current_step, env.n_steps - 1)
    df   = env.df

    rsi       = float(df.loc[step, "rsi_14"])       if "rsi_14"        in df.columns else 50.0
    macd_hist = float(df.loc[step, "macd_histogram"]) if "macd_histogram" in df.columns else 0.0
    dist50    = float(df.loc[step, "dist_ema_50"])   if "dist_ema_50"   in df.columns else 0.0

    holding = env.shares_held > 0

    # ------------------------------------------------------------------
    # TASK 1 — Easy: just make positive return, original logic is fine
    # ------------------------------------------------------------------
    if task_id == "task1":
        if not holding:
            if rsi < 70 and dist50 > -0.01 and macd_hist > -0.05:
                return 1  # BUY
        else:
            if rsi > 75 or dist50 < -0.02:
                return 2  # SELL
        return 0  # HOLD

    # ------------------------------------------------------------------
    # TASK 2 — Medium: protect Sharpe across 3 stocks
    #   Stricter entry: only buy confirmed uptrends
    #   Faster exit: lock in gains before reversals drag Sharpe down
    # ------------------------------------------------------------------
    if task_id == "task2":
        if not holding:
            # All 3 conditions must be clearly bullish
            if rsi < 55 and macd_hist > 0 and dist50 > 0:
                return 1  # BUY
        else:
            # Exit early to protect ratio — don't wait for deep reversals
            if rsi > 65 or dist50 < -0.015:
                return 2  # SELL
        return 0  # HOLD

    # ------------------------------------------------------------------
    # TASK 3 — Hard: survive volatile stock, drawdown must stay < 25%
    #   Only enter on very strong confirmed signals
    #   Exit at first sign of trouble OR if live drawdown > 8% from peak
    # ------------------------------------------------------------------
    if task_id == "task3":
        # Live drawdown check — hard stop-loss independent of indicators
        current_value = env._portfolio_value()
        peak_value    = env.peak_portfolio_value
        live_drawdown_pct = (current_value - peak_value) / (peak_value + 1e-8) * 100

        if holding:
            # Hard stop-loss: exit if we've lost 8% from our peak
            if live_drawdown_pct < -8.0:
                return 2  # SELL — capital protection
            # Normal exit conditions (tighter than task1)
            if rsi > 70 or dist50 < -0.01:
                return 2  # SELL
        else:
            # Only buy on strong confirmed uptrend signals
            if rsi < 45 and macd_hist > 0 and dist50 > 0:
                return 1  # BUY

        return 0  # HOLD

    # Fallback
    return 0


def run_episode(env: TradingEnv, sym: str, task_id: str = "task1", seed: int = SEED) -> dict:
    """Run one episode with the rule-based agent. Returns summary + portfolio_history."""
    obs, _info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    n_steps = 0

    while not done:
        action = get_action(env, sym, task_id=task_id)
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
    result = run_episode(env, TASK1_SYMBOL, task_id="task1")

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
        # task_id="task2" — stricter entry, faster exit per stock
        result = run_episode(env, symbol, task_id="task2")
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

    # Conservative reward: heavier drawdown penalty (matters for RL training context)
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
    # task_id="task3" — hard stop-loss at 8% live drawdown
    result = run_episode(env, TASK3_SYMBOL, task_id="task3")

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
    parser = argparse.ArgumentParser(description="AlphaTrader-RL Inference")
    parser.add_argument("--task1", type=str, default="TATASTEEL.NS", help="Symbol for Task 1")
    parser.add_argument("--task2", type=str, nargs="+", default=["TATASTEEL.NS", "GOLDBEES.NS", "SILVERBEES.NS"], help="Symbols for Task 2")
    parser.add_argument("--task3", type=str, default="YESBANK.NS", help="Symbol for Task 3")
    args = parser.parse_args()

    global TASK1_SYMBOL, TASK2_SYMBOLS, TASK3_SYMBOL
    TASK1_SYMBOL = args.task1
    TASK2_SYMBOLS = args.task2
    TASK3_SYMBOL = args.task3

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

    output_path = "inference_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results written to: {output_path}")

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