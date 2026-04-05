import json
import pandas as pd
from trading_env import TradingEnv
from inference import get_symbol_df, INITIAL_CAPITAL, PARQUET_PATH

df = pd.read_parquet(PARQUET_PATH)
SYMBOLS = ['TATASTEEL.NS', 'HINDALCO.NS', 'TATAPOWER.NS', 'YESBANK.NS']

for sym in SYMBOLS:
    sym_df = get_symbol_df(df, sym)
    env = TradingEnv(sym_df, initial_capital=INITIAL_CAPITAL)
    env.reset(seed=42)
    
    # Just hold
    done = False
    while not done:
        action = 1 if env.shares_held == 0 else 0
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    s = env.summary()
    print(f'{sym}: Buy&Hold Ret={s["total_return_pct"]:.2f}% Sharpe={s["sharpe_ratio"]:.2f} MaxDD={s["max_drawdown_pct"]:.2f}%')
