import json
import pandas as pd
from trading_env import TradingEnv
from inference import get_symbol_df, INITIAL_CAPITAL, PARQUET_PATH

df = pd.read_parquet(PARQUET_PATH)
SYMBOLS = ['TATASTEEL.NS', 'HINDALCO.NS', 'TATAPOWER.NS', 'YESBANK.NS']

def eval_agent(dist_ema_50_buy, dist_ema_20_buy, return_1d_sell):
    results = {}
    for sym in SYMBOLS:
        sym_df = get_symbol_df(df, sym)
        env = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
        env.reset(seed=42)
        done = False
        while not done:
            step = min(env.current_step, env.n_steps - 1)
            dist_ema_50 = env.df.loc[step, "dist_ema_50"]
            dist_ema_20 = env.df.loc[step, "dist_ema_20"]
            ret1d = env.df.loc[step, "return_1d"]
            
            holding = env.shares_held > 0
            
            buy_signal = dist_ema_50 > dist_ema_50_buy and dist_ema_20 > dist_ema_20_buy
            sell_signal = ret1d < return_1d_sell

            action = 0
            if not holding and buy_signal:
                action = 1
            elif holding and sell_signal:
                action = 2

            _, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        s = env.summary()
        results[sym] = s
    return results

best = None
best_score = -100

for d50 in [-0.05, 0.0, 0.02, 0.05]:
    for d20 in [-0.02, 0.0, 0.02, 0.05]:
        for rsell in [-0.05, -0.03, -0.02, -0.01, 0.0]:
            res = eval_agent(d50, d20, rsell)
            t1 = res['TATASTEEL.NS']['total_return_pct']
            sharpe_avg = (res['TATASTEEL.NS']['sharpe_ratio'] + res['HINDALCO.NS']['sharpe_ratio'] + res['TATAPOWER.NS']['sharpe_ratio']) / 3
            t3 = res['YESBANK.NS']['max_drawdown_pct']
            
            if t1 > 0 and sharpe_avg > 0.5 and (t3 > -25.0):
                print(f"PASS: d50>{d50}, d20>{d20}, sell<{rsell} -> T1:{t1:.2f}% S:{sharpe_avg:.2f} T3:{t3:.2f}%")
print("Done")
