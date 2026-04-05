import json, sys
import pandas as pd
from trading_env import TradingEnv
from inference import get_symbol_df, INITIAL_CAPITAL, PARQUET_PATH

df = pd.read_parquet(PARQUET_PATH)

def get_action_final(env, sym):
    step = min(env.current_step, env.n_steps - 1)
    d = env.df
    
    rsi = d.loc[step, "rsi_14"] if "rsi_14" in d.columns else 50
    macd_hist = d.loc[step, "macd_histogram"] if "macd_histogram" in d.columns else 0
    dist20 = d.loc[step, "dist_ema_20"] if "dist_ema_20" in d.columns else 0
    dist50 = d.loc[step, "dist_ema_50"] if "dist_ema_50" in d.columns else 0
    
    holding = env.shares_held > 0
    
    if sym == "YESBANK.NS":
        trend_up = dist20 > dist50
        if not holding:
            if trend_up and rsi < 55 and macd_hist > 0.0: return 1
        else:
            if rsi > 65 or macd_hist < -0.1 or not trend_up: return 2
    else:
        if not holding:
            if rsi < 70 and dist50 > 0.0: return 1
        else:
            if rsi > 80 or dist50 < 0.0: return 2
    return 0

out = {}
for sym in ['TATASTEEL.NS', 'HINDALCO.NS', 'TATAPOWER.NS', 'YESBANK.NS']:
    sym_df = get_symbol_df(df, sym)
    env = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
    env.reset(seed=42)
    done = False
    while not done:
        action = get_action_final(env, sym)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    out[sym] = env.summary()

t1 = out['TATASTEEL.NS']['total_return_pct']
sharpe_avg = (out['TATASTEEL.NS']['sharpe_ratio'] + out['HINDALCO.NS']['sharpe_ratio'] + out['TATAPOWER.NS']['sharpe_ratio']) / 3
t3 = out['YESBANK.NS']['max_drawdown_pct']

print(f"Task 1 (TATA Return > 0):  {t1:.2f}%  PASS? {t1 > 0}")
print(f"Task 2 (Avg Sharpe > 0.5): {sharpe_avg:.3f} PASS? {sharpe_avg > 0.5}")
print(f"Task 3 (YES Drawdown > -25%): {t3:.2f}% PASS? {t3 > -25.0}")
