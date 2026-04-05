import json, sys
import pandas as pd
from trading_env import TradingEnv
from inference import get_symbol_df, INITIAL_CAPITAL, PARQUET_PATH

df = pd.read_parquet(PARQUET_PATH)

def get_action_test(env, buy_rsi, sell_rsi, ema_dist):
    step = min(env.current_step, env.n_steps - 1)
    d = env.df
    
    rsi = d.loc[step, "rsi_14"]
    macd_hist = d.loc[step, "macd_histogram"]
    dist50 = d.loc[step, "dist_ema_50"]
    
    holding = env.shares_held > 0
    
    if not holding:
        if rsi < buy_rsi and dist50 > ema_dist:
            return 1
    else:
        if rsi > sell_rsi or dist50 < ema_dist:
            return 2
    return 0

for buy_rsi in [40, 50, 60, 70]:
    for sell_rsi in [60, 70, 80]:
        for ema_dist in [-0.05, 0.0, 0.05]:
            sharpes = []
            valid = True
            for sym in ['TATASTEEL.NS', 'HINDALCO.NS', 'TATAPOWER.NS']:
                sym_df = get_symbol_df(df, sym)
                env = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
                env.reset(seed=42)
                done = False
                while not done:
                    action = get_action_test(env, buy_rsi, sell_rsi, ema_dist)
                    _, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                s = env.summary()
                sharpes.append(s['sharpe_ratio'])
            avg_sharpe = sum(sharpes)/3.0
            if avg_sharpe > 0.50:
                print(f"PASS T2! buy={buy_rsi} sell={sell_rsi} ema={ema_dist} AvgSharpe={avg_sharpe:.3f}")
                
print("Search done")
