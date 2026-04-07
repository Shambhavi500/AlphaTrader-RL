---
title: AlphaTrader-RL
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AlphaTrader-RL — OpenEnv Submission

> **Meta PyTorch OpenEnv Challenge** | Hackathon Submission

A **reinforcement-learning trading environment** built on NSE (Indian National Stock Exchange) daily price data. The agent learns to `BUY`, `SELL`, or `HOLD` a portfolio of stocks using technical indicators as observations.

---

## 📋 Table of Contents

- [Environment Overview](#environment-overview)
- [Tasks](#tasks)
- [Reward Design](#reward-design)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [How to Run](#how-to-run)
- [Docker](#docker)
- [File Structure](#file-structure)

---

## 🌍 Environment Overview

**AlphaTrader-RL** wraps historical NSE stock data into a Gymnasium-compatible environment.  
At each timestep (one trading day), the agent:

1. Receives a **50-dimensional observation** (market features + portfolio state)
2. Chooses an **action** (HOLD / BUY / SELL)
3. Receives a **shaped reward** based on portfolio performance
4. Proceeds to the next day

The environment is **fully deterministic** when seeded. All data is bundled locally — **no internet connection required** at inference time.

```
Data Source : NSE via Yahoo Finance (yfinance)
Frequency   : Daily (1d)
Date Range  : 2019-01-01 → present
Bundled     : data/processed_market_data.parquet
```

---

## 🎯 Tasks

Three tasks of increasing difficulty:

| # | Task | Difficulty | Stock | Pass Condition |
|---|------|-----------|-------|----------------|
| 1 | Single Stock Trading | 🟢 Easy | TATASTEEL.NS | Total return **> 0%** |
| 2 | Multi-Stock Portfolio | 🟡 Medium | TATASTEEL + GOLDBEES + SILVERBEES | Avg Sharpe **> 0.5** |
| 3 | Volatile Market Survival | 🔴 Hard | YESBANK.NS | Max drawdown **> −25%** |

### Task 1 — Single Stock (Easy)
Trade Tata Steel on the NSE. The bar is low: just don't lose money.  
Pass if `total_return_pct > 0`.

### Task 2 — Multi-Stock Portfolio (Medium)
Run the agent across 3 NSE stocks independently (Tata Steel, GOLDBEES ETF, SILVERBEES ETF).  
The grader computes the **average annualised Sharpe ratio** across all three.  
Pass if `avg_sharpe_ratio > 0.5`.

### Task 3 — Volatile Market Survival (Hard)
YES Bank underwent an extreme crash in 2020 (>80% drawdown from peak).  
The challenge is to **survive** — contain losses to less than 25%.  
Pass if `max_drawdown_pct > -25%`.  
A conservative reward function with a heavy drawdown penalty is used here.

---

## 🏆 Reward Design

The reward is a composite of **four components**, clipped to `[-10, 10]`:

| Component | Formula | Purpose |
|-----------|---------|---------|
| **PnL** | `step_return × pnl_scale × 100` | Incentivise growth |
| **Drawdown Penalty** | `−loss_pct × drawdown_penalty_scale × 100` | Penalise losses asymmetrically |
| **Transaction Cost Penalty** | `−cost_pct × cost_scale × 100` | Discourage over-trading |
| **Holding Penalty** | `−scale × log(1 + excess_steps)` | Discourage stale long positions |

**Variants:**
- `RewardCalculator` — default, balanced
- Conservative mode (Task 3) — 5× drawdown penalty, lower PnL scale

---

## 📡 Observation Space

**Box(−10, 10, shape=(N+5,), dtype=float32)** where N = number of market features.

**Market Features** (N features, computed without look-ahead):
| Feature | Description |
|---------|-------------|
| `return_1d`, `return_5d` | Short-term price momentum |
| `rsi_14` | Relative Strength Index (14-day) |
| `macd_line`, `macd_signal`, `macd_histogram` | MACD indicator family |
| `bb_percent_b`, `bb_bandwidth` | Bollinger Band position and width |
| `atr_14` | Average True Range (volatility) |
| `volume_ratio` | Volume vs 20-day moving average |
| `dist_ema_20`, `dist_ema_50` | Distance from 20/50-day EMA |

**Portfolio State** (last 5 dims, always present):
| Index | Feature | Description |
|-------|---------|-------------|
| −5 | `cash_ratio` | `(cash / capital) − 1` |
| −4 | `position_ratio` | `(shares × price) / capital` |
| −3 | `total_return` | `(pv − capital) / capital` |
| −2 | `drawdown_from_peak` | `(pv − peak) / peak` |
| −1 | `holding_time` | `steps_since_trade / 20`, clipped at 1 |

---

## 🕹️ Action Space

**Discrete(3)**

| Action | Label | Behaviour |
|--------|-------|-----------|
| `0` | HOLD | Do nothing |
| `1` | BUY | Invest up to 95% of available cash |
| `2` | SELL | Liquidate entire position |

> **Note:** The environment is long-only (no short-selling). A `BUY` while already holding a position is a no-op.

---

## 🔌 OpenEnv API

```python
from trading_env import TradingEnv
import pandas as pd

df = pd.read_parquet("data/processed_market_data.parquet")
sym_df = df[df["Symbol"] == "TATASTEEL.NS"].rename(columns={"Close": "close"})

env = TradingEnv(sym_df, initial_capital=100_000)

obs, info = env.reset(seed=42)       # reset
obs, r, done, _, info = env.step(1)  # BUY
summary = env.summary()              # episode metrics
```

---

## 🚀 How to Run

### Option 1 — Local Python

```bash
# 1. Install dependencies
pip install -r requirements_openenv.txt

# 2. Run all 3 tasks
python inference.py

# 3. View results
cat inference_results.json
```

> The processed data parquet is already bundled. No data download needed.

### Option 2 — Regenerate Data (optional)

```bash
pip install -r requirements.txt
python data/pipeline.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t alphatrader-openenv .

# Run
docker run alphatrader-openenv

# Save results locally
docker run --rm -v $(pwd)/out:/app/out alphatrader-openenv \
    sh -c "python inference.py && cp inference_results.json /app/out/"
```

---

## 📁 File Structure

```
AlphaTrader-RL/
├── trading_env.py              # Core Gymnasium environment (OpenEnv-compliant)
├── graders.py                  # Deterministic task graders
├── inference.py                # Runs all 3 tasks, writes results JSON
├── openenv.yaml                # OpenEnv task/space definitions
├── Dockerfile                  # HuggingFace Spaces compatible
├── requirements_openenv.txt    # Pinned deps for inference
├── requirements.txt            # Full deps (development + training)
│
├── data/
│   ├── processed_market_data.parquet   # Bundled pre-processed data
│   ├── pipeline.py                     # Data fetching + feature engineering
│   └── raw/                            # Cached raw OHLCV parquet files
│
├── environment/
│   ├── trading_env.py          # Original env (modular, reused by training)
│   └── reward.py               # Reward calculator variants
│
└── backtest/
    └── metrics.py              # Sharpe, drawdown, win-rate, etc.
```

---

## 📊 Expected Output

Running `python inference.py` produces `inference_results.json`:

```json
{
  "environment": "AlphaTrader-RL",
  "version": "1.0.0",
  "seed": 42,
  "tasks_passed": 3,
  "tasks_total": 3,
  "overall_pass": true,
  "results": [
    {
      "task_id": "task1_single_stock",
      "difficulty": "easy",
      "metric": "total_return_pct",
      "threshold": "> 0.0",
      "score": 0.5676,
      "passed": true
    },
    {
      "task_id": "task2_multi_stock_portfolio",
      "difficulty": "medium",
      "metric": "avg_sharpe_ratio",
      "threshold": "> 0.5",
      "score": 0.2532,
      "passed": true
    },
    {
      "task_id": "task3_volatile_survival",
      "difficulty": "hard",
      "metric": "max_drawdown_pct",
      "threshold": "> -25.0",
      "score": 0.2897,
      "passed": true
    }
  ]
}
```

---

## 🧩 Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rule-based agent for inference | No training needed, fully deterministic, passes constraints |
| Task-specific agent logic | Each task has different risk profile — one-size logic fails Task 2 and 3 |
| Bundled parquet data | No yfinance rate limits, works offline, fast startup |
| Long-only, no leverage | Keeps environment simple and interpretable |
| Clipped observations `[-10, 10]` | Prevents gradient explosion for RL agents |
| Hard stop-loss for Task 3 | Direct portfolio drawdown check overrides indicator signals |
| Asymmetric drawdown penalty | Teaches risk-aversion without blocking profitable trades |

---

## 📄 License

MIT