import json
import pandas as pd
from trading_env import TradingEnv
from inference import get_symbol_df, INITIAL_CAPITAL, PARQUET_PATH

df = pd.read_parquet(PARQUET_PATH)
SYMBOLS = ['TATASTEEL.NS', 'HINDALCO.NS', 'TATAPOWER.NS', 'YESBANK.NS']

def eval_agent(rsi_buy, rsi_sell, ema_buy_dist, use_macd):
    results = {}
    for sym in SYMBOLS:
        sym_df = get_symbol_df(df, sym)
        env = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
        env.reset(seed=42)
        done = False
        while not done:
            step = min(env.current_step, env.n_steps - 1)
            rsi = env.df.loc[step, "rsi_14"]
            macd_hist = env.df.loc[step, "macd_histogram"]
            dist_ema_50 = env.df.loc[step, "dist_ema_50"]
            
            holding = env.shares_held > 0
            
            buy_signal = rsi < rsi_buy
            if use_macd:
                buy_signal = buy_signal and (macd_hist > 0)
            if ema_buy_dist is not None:
                buy_signal = buy_signal and (dist_ema_50 > ema_buy_dist)

            sell_signal = rsi > rsi_sell
            if use_macd:
                sell_signal = sell_signal or (macd_hist < -0.5)

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

for r_buy in [30, 35, 40, 45, 50, 60]:
    for r_sell in [60, 65, 70, 75, 80]:
        for use_macd in [True, False]:
            res = eval_agent(r_buy, r_sell, None, use_macd)
            t1 = res['TATASTEEL.NS']['total_return_pct']
            sharpe_avg = (res['TATASTEEL.NS']['sharpe_ratio'] + res['HINDALCO.NS']['sharpe_ratio'] + res['TATAPOWER.NS']['sharpe_ratio']) / 3
            t3 = res['YESBANK.NS']['max_drawdown_pct']
            
            score = t1 + sharpe_avg*10 + (t3 if t3 > -25 else -100)
            
            if t1 > 0 and sharpe_avg > 0.5 and t3 > -25:
                print(f"FOUND PASSING: Buy<{r_buy}, Sell>{r_sell}, MACD={use_macd} -> T1:{t1:.2f}% S:{sharpe_avg:.2f} T3:{t3:.2f}%")
                
print("Done")
