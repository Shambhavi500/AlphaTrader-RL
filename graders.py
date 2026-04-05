"""
AlphaTrader-RL | OpenEnv Graders
=================================
Deterministic pass/fail graders for each of the 3 tasks.

Task 1 (Easy)   — Single stock, TATASTEEL.NS
  PASS if total_return_pct > 0

Task 2 (Medium) — Multi-stock portfolio (equal-weight averaging)
  PASS if annualised Sharpe ratio > 0.5

Task 3 (Hard)   — Volatile stock survival (YESBANK.NS — highly volatile)
  PASS if max_drawdown_pct > -25  (i.e. drawdown less than 25%)
"""
import numpy as np
import pandas as pd
from typing import Any


def _sharpe(portfolio_history: list) -> float:
    ph = pd.Series(portfolio_history, dtype=float)
    returns = ph.pct_change().dropna()
    if returns.std() < 1e-10:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _max_drawdown_pct(portfolio_history: list) -> float:
    ph = pd.Series(portfolio_history, dtype=float)
    dd = (ph - ph.cummax()) / ph.cummax()
    return float(dd.min() * 100)


def _total_return_pct(portfolio_history: list, initial_capital: float) -> float:
    return (portfolio_history[-1] - initial_capital) / initial_capital * 100


# ---------------------------------------------------------------------------
# Grader functions
# ---------------------------------------------------------------------------

def grade_task1(portfolio_history: list, initial_capital: float) -> dict:
    """
    Task 1 — Single Stock Easy
    PASS: total return > 0%
    """
    ret = _total_return_pct(portfolio_history, initial_capital)
    passed = bool(ret > 0.0)
    
    # Normalize return % to [0, 1]
    score = min(max(ret / 100.0, 0.0), 1.0)
    
    return {
        "task_id": "task1_single_stock",
        "difficulty": "easy",
        "metric": "total_return_pct",
        "threshold": "> 0.0",
        "score": round(score, 4),
        "passed": passed,
        "message": f"Return {ret:.2f}% — {'PASS' if passed else 'FAIL'}",
    }


def grade_task2(portfolio_histories: list, initial_capital: float) -> dict:
    """
    Task 2 — Multi-Stock Portfolio (Medium)
    portfolio_histories: list of per-stock portfolio_history lists
    PASS: average annualised Sharpe ratio > 0.5
    """
    sharpes = [_sharpe(ph) for ph in portfolio_histories]
    avg_sharpe = float(np.mean(sharpes))
    passed = bool(avg_sharpe > 0.5)
    
    # Normalize Sharpe ratio
    score = min(max(avg_sharpe / 2.0, 0.0), 1.0)
    
    return {
        "task_id": "task2_multi_stock_portfolio",
        "difficulty": "medium",
        "metric": "avg_sharpe_ratio",
        "threshold": "> 0.5",
        "score": round(score, 4),
        "passed": passed,
        "message": f"Avg Sharpe {avg_sharpe:.4f} — {'PASS' if passed else 'FAIL'}",
    }


def grade_task3(portfolio_history: list, initial_capital: float) -> dict:
    """
    Task 3 — Volatile Market Survival (Hard)
    PASS: max drawdown > -25% (i.e. loss is contained)
    """
    mdd = _max_drawdown_pct(portfolio_history)
    passed = bool(mdd > -25.0)
    
    # Lower drawdown = better, normalize between 0 and 1
    score = 1.0 - min(abs(mdd) / 25.0, 1.0) if mdd < 0 else 1.0
    
    return {
        "task_id": "task3_volatile_survival",
        "difficulty": "hard",
        "metric": "max_drawdown_pct",
        "threshold": "> -25.0",
        "score": round(score, 4),
        "passed": passed,
        "message": f"Max Drawdown {mdd:.2f}% — {'PASS' if passed else 'FAIL'}",
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic portfolio histories for smoke-testing
    growing = [100_000 * (1 + 0.001) ** i for i in range(252)]
    flat    = [100_000] * 252
    crash   = [100_000 * max(0.5, 1 - 0.001 * i) for i in range(252)]

    r1 = grade_task1(growing, 100_000)
    r2 = grade_task2([growing, flat], 100_000)
    r3 = grade_task3(crash, 100_000)

    for r in [r1, r2, r3]:
        print(r)

    print("\n✅ Graders smoke test passed.")
