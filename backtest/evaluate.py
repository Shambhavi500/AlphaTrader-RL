"""
AlphaTrader-RL | Part 4: Backtesting Evaluator

Usage
-----
    cd d:\\AlphaTrader-RL
    python backtest/evaluate.py [--symbol TATASTEEL.NS] [--model models/best_model]
                                [--fold-idx -1]  # -1 = last (most recent) fold

What it does
------------
1. Loads processed parquet and builds walk-forward folds (same parameters as training).
2. Scales features of the chosen test fold (fit on the matching train split — no leakage).
3. Loads the saved PPO model and runs ONE deterministic episode on the test fold.
4. Collects a detailed trade log from the episode.
5. Computes BacktestMetrics and prints a clean summary table.
6. Plots the equity curve vs. Buy-and-Hold and saves to logs/backtest_equity_curve.png.
"""
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import WalkForwardPreprocessor
from environment.trading_env import TradingEnv
from agent.ppo_agent import PPOAgent
from backtest.metrics import BacktestMetrics

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/backtest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Backtest")

DATA_PATH   = "data/processed_market_data.parquet"
PLOT_PATH   = "logs/backtest_equity_curve.png"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (mirrors agent/train.py to guarantee identical preprocessing)
# ──────────────────────────────────────────────────────────────────────────────

def _load_and_filter(symbol: str) -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at '{DATA_PATH}'. "
            "Run `python data/pipeline.py` first."
        )
    df = pd.read_parquet(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" in df.columns:
        if symbol not in df["Symbol"].unique():
            avail = sorted(df["Symbol"].unique().tolist())
            raise ValueError(f"Symbol '{symbol}' not found. Available: {avail}")
        df = df[df["Symbol"] == symbol].copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df.sort_index()


def _build_env_df(
    scaled_array: np.ndarray,
    feature_names: list,
    raw_close: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(scaled_array, columns=feature_names)
    df["close"] = raw_close.values
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner with trade-log capture
# ──────────────────────────────────────────────────────────────────────────────

def _run_episode(agent: PPOAgent, env: TradingEnv) -> list:
    """
    Run one deterministic episode and return a detailed trade log.

    Returns
    -------
    list[dict]  — one entry per executed trade (BUY / SELL only).
    """
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
    obs, _ = env.reset()
    done = False
    trade_log = []
    prev_shares = 0

    while not done:
        action_arr, _ = agent.model.predict(obs, deterministic=True)
        action = int(action_arr)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        shares_now = info["shares_held"]
        if action in (1, 2) and shares_now != prev_shares:
            trade_log.append({
                "step":            info["current_step"],
                "action":          ACTION_NAMES[action],
                "price":           info["current_price"],
                "shares":          abs(shares_now - prev_shares),
                "portfolio_value": info["portfolio_value"],
                "cost":            0.0,        # approximate — env subtracts from cash
            })
        prev_shares = shares_now

    return trade_log


# ──────────────────────────────────────────────────────────────────────────────
# Equity curve plot
# ──────────────────────────────────────────────────────────────────────────────

def _plot_equity_curve(
    portfolio_history: list,
    bah_history: list,
    date_index: pd.DatetimeIndex,
    symbol: str,
    metrics_summary: dict,
    save_path: str,
):
    """Save a dual-panel equity-curve chart."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"AlphaTrader-RL | Backtest — {symbol}\n"
        f"Sharpe: {metrics_summary['sharpe_ratio']}  |  "
        f"Return: {metrics_summary['total_return_pct']:+.2f}%  |  "
        f"Alpha vs B&H: {metrics_summary['alpha_pct']:+.2f}%",
        fontsize=13, fontweight="bold", y=0.98,
    )

    pv = pd.Series(portfolio_history)
    bah = pd.Series(bah_history)

    # Use dates if available, else integer index
    xs = date_index[:len(pv)] if date_index is not None and len(date_index) >= len(pv) else np.arange(len(pv))

    # ── Panel 1: Equity curves ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(xs, pv.values,   color="#00C4A7", linewidth=1.8, label="PPO Agent")
    ax1.plot(xs, bah.values,  color="#FF6B6B", linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax1.fill_between(xs, pv.values, bah.values,
                     where=(pv.values >= bah.values), alpha=0.15, color="#00C4A7", label="Alpha region")
    ax1.fill_between(xs, pv.values, bah.values,
                     where=(pv.values <  bah.values), alpha=0.15, color="#FF6B6B")
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("")

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────
    ax2 = axes[1]
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max * 100
    ax2.fill_between(xs, drawdown.values, 0, color="#FF6B6B", alpha=0.6, label="Drawdown %")
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Date" if date_index is not None else "Step", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    if date_index is not None and len(date_index) >= len(pv):
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve saved to %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Buy-and-Hold portfolio series
# ──────────────────────────────────────────────────────────────────────────────

def _bah_series(prices: pd.Series, initial_capital: float) -> list:
    """Simulate Buy-and-Hold: buy at step-0 price, hold to end."""
    p0 = float(prices.iloc[0])
    shares = int(initial_capital * 0.95 / p0)
    cash_remaining = initial_capital - shares * p0
    return [cash_remaining + shares * float(p) for p in prices]


# ──────────────────────────────────────────────────────────────────────────────
# Main backtest function
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str = "TATASTEEL.NS",
    model_path: str = "models/best_model",
    fold_idx: int = -1,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Load model + data, run one deterministic test episode, compute and print metrics.

    Parameters
    ----------
    symbol        : NSE ticker to evaluate.
    model_path    : Path to saved model (.zip suffix added automatically).
    fold_idx      : Which walk-forward fold's test split to use. -1 = last fold.
    initial_capital : Starting portfolio value (₹).

    Returns
    -------
    dict : metrics summary.
    """
    print("=" * 65)
    print("  AlphaTrader-RL | Part 4: Backtesting Engine")
    print(f"  Symbol: {symbol}  |  Model: {model_path}.zip")
    print("=" * 65)

    # ── 1. Load & filter data ────────────────────────────────────────────────
    df = _load_and_filter(symbol)
    close_col = "Close" if "Close" in df.columns else "close"

    # ── 2. Build folds (identical settings to training) ─────────────────────
    preprocessor = WalkForwardPreprocessor(train_years=3.0, val_months=6, test_months=6)
    folds = preprocessor.create_folds(df)

    if not folds:
        raise RuntimeError(
            "No folds generated. Ensure data spans ≥ 4 years. "
            "Run `python data/pipeline.py` first."
        )

    chosen = fold_idx % len(folds)          # supports -1 → last fold
    train_df, val_df, test_df = folds[chosen]

    print(f"\n  Using fold {chosen + 1}/{len(folds)}")
    print(f"  Train : {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} rows)")
    print(f"  Val   : {val_df.index[0].date()} → {val_df.index[-1].date()} ({len(val_df)} rows)")
    print(f"  Test  : {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df)} rows)\n")
    logger.info("Backtest fold %d: test %s → %s (%d rows)",
                chosen + 1, test_df.index[0].date(), test_df.index[-1].date(), len(test_df))

    # ── 3. Scale features (fit on train — NO leakage) ────────────────────────
    X_train, _X_val, X_test, feature_names = preprocessor.prepare_fold_data(
        train_df, val_df, test_df
    )

    raw_close_test = test_df[close_col].reset_index(drop=True)
    test_env_df    = _build_env_df(X_test, feature_names, raw_close_test)
    test_dates     = pd.to_datetime(test_df.index)

    # ── 4. Create environment & load model ──────────────────────────────────
    test_env = TradingEnv(test_env_df, initial_capital=initial_capital)

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found at '{model_path}.zip'. "
            "Run `python agent/train.py` first to train and save a model."
        )
    agent = PPOAgent.load(model_path, env=test_env)
    print("  [✓] Model loaded successfully.\n")

    # ── 5. Run one deterministic episode ────────────────────────────────────
    print("  Running deterministic backtest episode …")
    trade_log = _run_episode(agent, test_env)

    portfolio_history = test_env.portfolio_history         # collected by env
    prices_series     = raw_close_test.iloc[: len(portfolio_history)]
    bah_history       = _bah_series(prices_series, initial_capital)

    # ── 6. Compute metrics ──────────────────────────────────────────────────
    bmetrics = BacktestMetrics(
        portfolio_history=portfolio_history,
        trade_log=trade_log,
        initial_capital=initial_capital,
        prices=prices_series.tolist(),
    )
    summary = bmetrics.summary()
    logger.info("Backtest summary: %s", summary)

    # ── 7. Print summary table ──────────────────────────────────────────────
    _print_summary(summary, symbol)

    # ── 8. Plot equity curve ────────────────────────────────────────────────
    _plot_equity_curve(
        portfolio_history=portfolio_history,
        bah_history=bah_history,
        date_index=test_dates,
        symbol=symbol,
        metrics_summary=summary,
        save_path=PLOT_PATH,
    )
    print(f"\n  [✓] Equity curve saved → {PLOT_PATH}")
    print("=" * 65)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(summary: dict, symbol: str):
    win = "✅" if summary["total_return_pct"] > summary["buy_hold_return_pct"] else "⚠️"

    rows = [
        ("Metric",                          "Agent",                              "Buy & Hold"),
        ("─" * 28,                          "─" * 14,                             "─" * 14),
        ("Total Return (%)",                f"{summary['total_return_pct']:+.2f}", f"{summary['buy_hold_return_pct']:+.2f}"),
        ("Alpha over B&H (%)",              f"{summary['alpha_pct']:+.2f}",        "—"),
        ("Max Drawdown (%)",                f"{summary['max_drawdown_pct']:.2f}",  "—"),
        ("Sharpe Ratio",                    f"{summary['sharpe_ratio']:.3f}",      "—"),
        ("Sortino Ratio",                   f"{summary['sortino_ratio']:.3f}",     "—"),
        ("Calmar Ratio",                    f"{summary['calmar_ratio']:.3f}",      "—"),
        ("Number of Trades",                f"{summary['num_trades']}",            "1 (buy) + 1 (sell)"),
        ("Win Rate (%)",                    f"{summary['win_rate_pct']:.1f}",      "—"),
        ("Avg Trade PnL (%)",               f"{summary['avg_trade_pct']:+.3f}",    "—"),
        ("Final Portfolio Value (₹)",       f"₹{summary['final_portfolio_value']:,.2f}", "—"),
    ]

    print(f"\n  {'─' * 58}")
    print(f"  Backtest Summary — {symbol}  {win}")
    print(f"  {'─' * 58}")
    for r in rows:
        print(f"  {r[0]:<30} {r[1]:<18} {r[2]}")
    print(f"  {'─' * 58}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="AlphaTrader-RL Backtesting Engine")
    p.add_argument("--symbol",  default="TATASTEEL.NS",    help="NSE symbol (default: TATASTEEL.NS)")
    p.add_argument("--model",   default="models/best_model", help="Path to saved model (no .zip suffix)")
    p.add_argument("--fold-idx", type=int, default=-1,     help="Walk-forward fold index to test (-1 = last)")
    p.add_argument("--capital", type=float, default=100_000.0, help="Initial capital in ₹ (default: 100000)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        symbol=args.symbol,
        model_path=args.model,
        fold_idx=args.fold_idx,
        initial_capital=args.capital,
    )
