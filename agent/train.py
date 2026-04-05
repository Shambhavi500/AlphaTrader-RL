"""
AlphaTrader-RL | Part 3: Walk-Forward Training Script

Usage
-----
    cd d:\\AlphaTrader-RL
    python agent/train.py [--symbol TATASTEEL.NS] [--timesteps 200000] [--eval-episodes 3]

Walk-forward logic
------------------
For each chronological fold produced by WalkForwardPreprocessor:
  1. Scale features (scaler fit ONLY on train to prevent leakage).
  2. Re-attach unscaled `close` prices so TradingEnv can compute real P&L.
  3. Train PPOAgent on the train split.
  4. Evaluate on the val split → record Sharpe.
  5. If best Sharpe so far, overwrite models/best_model.zip.
  6. Early-stop if no improvement for PATIENCE consecutive evaluations.
After all folds, run final evaluation on the test split of the best fold.
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import WalkForwardPreprocessor
from environment.trading_env import TradingEnv
from agent.ppo_agent import PPOAgent

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Train")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PATIENCE = 5          # early-stop after N non-improving evals
DATA_PATH = "data/processed_market_data.parquet"
MODEL_SAVE_PATH = "models/best_model"


def _build_env_df(
    scaled_array: np.ndarray,
    feature_names: list,
    raw_close: pd.Series,
) -> pd.DataFrame:
    """
    Combines scaled feature columns with the unscaled `close` price column.

    Parameters
    ----------
    scaled_array : np.ndarray  shape (n, n_features)
    feature_names : list[str]  column names matching scaled_array columns
    raw_close : pd.Series      unscaled close prices (same length)

    Returns
    -------
    pd.DataFrame with columns = feature_names + ["close"]
    """
    df = pd.DataFrame(scaled_array, columns=feature_names)
    df["close"] = raw_close.values
    return df


def _load_and_filter(symbol: str) -> pd.DataFrame:
    """Load parquet and filter to a single ticker, keeping Date as an index."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at '{DATA_PATH}'. "
            "Please run `python data/pipeline.py` first."
        )
    df = pd.read_parquet(DATA_PATH)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    if "Symbol" in df.columns:
        if symbol not in df["Symbol"].unique():
            available = sorted(df["Symbol"].unique().tolist())
            raise ValueError(f"Symbol '{symbol}' not in data. Available: {available}")
        df = df[df["Symbol"] == symbol].copy()

    if "Date" in df.columns:
        df = df.set_index("Date")

    df = df.sort_index()
    logger.info("Loaded %d rows for symbol '%s'.", len(df), symbol)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(symbol: str = "TATASTEEL.NS", total_timesteps: int = 200_000, eval_episodes: int = 3):
    print("=" * 65)
    print(f"  AlphaTrader-RL | PPO Walk-Forward Training")
    print(f"  Symbol: {symbol}  |  Timesteps/fold: {total_timesteps:,}")
    print("=" * 65)

    # 1. Load data
    df = _load_and_filter(symbol)

    # Identify the raw Close column (pipeline saves as 'Close')
    close_col = "Close" if "Close" in df.columns else "close"

    # 2. Walk-forward folds
    preprocessor = WalkForwardPreprocessor(train_years=3.0, val_months=6, test_months=6)
    folds = preprocessor.create_folds(df)

    if not folds:
        print("\n  [INFO] Data span < 4 years. Applying flexible 70/15/15 split fallback.")
        if len(df) < 200:
            print(f"[ERROR] Insufficient data. Found {len(df)} rows, minimum required is 200.")
            logger.error("Insufficient data for %s: %d rows.", symbol, len(df))
            return
        
        # Fallback: 70% Train, 15% Val, 15% Test
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        folds = [(train_df, val_df, test_df)]
        print(f"  [Fallback] Single fold created: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

    print(f"\n  {len(folds)} fold(s) generated.\n")

    best_sharpe = -np.inf
    best_fold_idx = 0
    best_agent: PPOAgent = None
    best_test_df_info: dict = {}   # {scaled_array, feature_names, raw_close}
    no_improve_count = 0

    for fold_idx, (train_df, val_df, test_df) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"─── Fold {fold_num}/{len(folds)} " + "─" * 45)
        print(f"    Train : {train_df.index[0].date()} → {train_df.index[-1].date()}  ({len(train_df)} rows)")
        print(f"    Val   : {val_df.index[0].date()} → {val_df.index[-1].date()}  ({len(val_df)} rows)")
        print(f"    Test  : {test_df.index[0].date()} → {test_df.index[-1].date()}  ({len(test_df)} rows)")

        # 3. Scale features (fit on train only)
        try:
            X_train, X_val, X_test, feature_names = preprocessor.prepare_fold_data(
                train_df, val_df, test_df
            )
        except AssertionError as exc:
            print(f"    [SKIP] Scaling assertion failed: {exc}")
            logger.warning("Fold %d skipped due to scaling error: %s", fold_num, exc)
            continue

        # 4. Preserve unscaled close prices
        raw_close_train = train_df[close_col].reset_index(drop=True)
        raw_close_val = val_df[close_col].reset_index(drop=True)
        raw_close_test = test_df[close_col].reset_index(drop=True)

        # 5. Build DataFrames for TradingEnv
        train_env_df = _build_env_df(X_train, feature_names, raw_close_train)
        val_env_df = _build_env_df(X_val, feature_names, raw_close_val)
        test_env_df = _build_env_df(X_test, feature_names, raw_close_test)

        obs_dim = len(feature_names) + 5  # +5 portfolio state features
        print(f"    Obs dims  : {obs_dim}  |  Features: {len(feature_names)}")

        # 6. Train
        train_env = TradingEnv(train_env_df)
        agent = PPOAgent(train_env, tensorboard_log="logs/tensorboard", verbose=0)
        print(f"\n  [Fold {fold_num}] Training for {total_timesteps:,} timesteps …")
        agent.train(
            total_timesteps=total_timesteps,
            tb_log_name=f"PPO_fold{fold_num}_{symbol.replace('.', '_')}",
        )

        # 7. Evaluate on validation split
        print(f"\n  [Fold {fold_num}] Validation episodes:")
        val_env = TradingEnv(val_env_df)
        fold_sharpe = agent.evaluate(val_env, n_episodes=eval_episodes)
        print(f"  → Mean Val Sharpe: {fold_sharpe:.4f}")
        logger.info("Fold %d | Val Sharpe=%.4f", fold_num, fold_sharpe)

        # 8. Track best & early stopping
        if fold_sharpe > best_sharpe:
            best_sharpe = fold_sharpe
            best_fold_idx = fold_idx
            best_agent = agent
            best_test_df_info = {
                "scaled_array": X_test,
                "feature_names": feature_names,
                "raw_close": raw_close_test,
            }
            symbol_model_path = f"models/{symbol}_model"
            agent.save(symbol_model_path)
            agent.save(MODEL_SAVE_PATH)  # Also save as best_model for backward compatibility
            logger.info("New best Sharpe=%.4f at fold %d — model saved to %s and %s.", 
                        best_sharpe, fold_num, symbol_model_path, MODEL_SAVE_PATH)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  [Early-stop counter: {no_improve_count}/{PATIENCE}]")
            if no_improve_count >= PATIENCE:
                print(f"\n  [Early Stop] No improvement for {PATIENCE} folds. Stopping.\n")
                logger.info("Early stopping triggered at fold %d.", fold_num)
                break

        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Final test evaluation on the best fold
    # ──────────────────────────────────────────────────────────────────────────
    if best_agent is None:
        print("[ERROR] No model was trained successfully. Exiting.")
        return

    print("=" * 65)
    print(f"  Best fold: {best_fold_idx + 1}  |  Best Val Sharpe: {best_sharpe:.4f}")
    print(f"  Running final TEST evaluation …\n")

    test_env_df_final = _build_env_df(
        best_test_df_info["scaled_array"],
        best_test_df_info["feature_names"],
        best_test_df_info["raw_close"],
    )
    test_env = TradingEnv(test_env_df_final)
    print("  [Test] Episode results:")
    test_sharpe = best_agent.evaluate(test_env, n_episodes=eval_episodes)

    # Print full summary of last test episode
    print("\n" + "─" * 65)
    print("  Final Test Summary (env.summary()):")
    test_summary = test_env.summary()
    for k, v in test_summary.items():
        print(f"    {k:30s}: {v}")
    print("─" * 65)
    print(f"  Mean Test Sharpe  : {test_sharpe:.4f}")
    print(f"  Best model saved  : {MODEL_SAVE_PATH}.zip")
    print("=" * 65)
    logger.info("Final test Sharpe=%.4f | Summary: %s", test_sharpe, test_summary)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="AlphaTrader-RL Walk-Forward PPO Trainer")
    p.add_argument("--symbol", default="TATASTEEL.NS", help="NSE symbol to train on (default: TATASTEEL.NS)")
    p.add_argument("--timesteps", type=int, default=200_000, help="PPO timesteps per fold (default: 200000)")
    p.add_argument("--eval-episodes", type=int, default=3, help="Eval episodes per fold (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        symbol=args.symbol,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
    )
