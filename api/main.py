"""
AlphaTrader-RL | Part 6: FastAPI Backend
Provides endpoints for backtesting, paper trading, and AI explanations.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backtest.evaluate import run_backtest
from api.paper_trader import PaperTrader
from environment.trading_env import TradingEnv
import pandas as pd
import numpy as np
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaAPI")

app = FastAPI(title="AlphaTrader-RL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Request/Response Schemas
# ──────────────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol: str = "TATASTEEL.NS"
    model_path: str = "models/best_model"
    initial_capital: float = 100_000.0

class TradeRequest(BaseModel):
    symbol: str = "TATASTEEL.NS"
    # In a real app, features would come from a live data feed
    # Here we allow passing them for demonstration
    features: Dict[str, float]
    current_price: float
    observation: List[float]

# ──────────────────────────────────────────────────────────────────────────────
# App State (Simplified)
# ──────────────────────────────────────────────────────────────────────────────
# In a real app, this might be a database or a more robust state manager
class AppState:
    def __init__(self):
        self.traders: Dict[str, PaperTrader] = {}

state = AppState()

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AlphaTrader-RL API is online", "status": "healthy"}

@app.post("/backtest")
async def execute_backtest(req: BacktestRequest):
    """
    Triggers a backtest for a specific symbol and returns the summary metrics.
    """
    try:
        summary = run_backtest(
            symbol=req.symbol,
            model_path=req.model_path,
            initial_capital=req.initial_capital
        )
        return {"status": "success", "metrics": summary}
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{symbol}")
async def get_portfolio(symbol: str):
    """Returns the current state of a paper trader."""
    if symbol not in state.traders:
        raise HTTPException(status_code=404, detail="Trader not initialized for this symbol.")
    return state.traders[symbol].get_status()

@app.post("/trade")
async def post_trade(req: TradeRequest):
    """
    Executes a single paper trade step based on provided observation and price.
    Returns the action taken and the LLM explanation.
    """
    try:
        if req.symbol not in state.traders:
            # Lazy initialize the trader
            model_p = "models/best_model"
            if not os.path.exists(model_p + ".zip"):
                model_p = "models/smoke_test_model"
                logger.info(f"best_model.zip missing, falling back to {model_p}")
                
            trader = PaperTrader(model_path=model_p, symbol=req.symbol)
            
            # Ensure dummy_df results in 41-dim observation (36 market features + 5 portfolio)
            # We pad it to exactly 36 feature columns + 1 close column
            dummy_cols = ["close"] + [f"feat_{i}" for i in range(36)]
            dummy_data = {col: [0.0] for col in dummy_cols}
            dummy_data["close"] = [req.current_price]
            
            # Overlay specific features from the request if they exist in our set
            for i, (k, v) in enumerate(req.features.items()):
                if i < 36:
                    dummy_data[f"feat_{i}"] = [v]
            
            dummy_df = pd.DataFrame(dummy_data)
            env = TradingEnv(dummy_df)
            trader.load_agent(env)
            state.traders[req.symbol] = trader
        
        trader = state.traders[req.symbol]
        obs_array = np.array(req.observation, dtype=np.float32)
        
        result = trader.execute_step(
            observation=obs_array,
            price=req.current_price,
            feature_dict=req.features
        )
        logger.info(f"Trade executed for {req.symbol}: action={result['action']}, has_explanation={bool(result.get('explanation'))}")
        return result
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signal/batch")
async def get_batch_signals(symbols: str):
    """
    Returns signals for multiple symbols at once.
    symbols: comma-separated string (e.g. ?symbols=TATASTEEL.NS,RELIANCE.NS)
    """
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results = []
    for sym in symbol_list:
        try:
            sig = await generate_signal(sym)
            results.append(sig)
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return results

@app.get("/signal/{symbol}")
async def get_signal(symbol: str):
    """
    Returns a dynamic signal for ANY NSE symbol.
    """
    try:
        return await generate_signal(symbol)
    except Exception as e:
        logger.error(f"Signal failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routes")
async def list_routes():
    """Returns all registered application routes."""
    url_list = [{"path": route.path, "name": route.name, "methods": list(route.methods)} 
                for route in app.routes]
    return url_list

@app.get("/price/{symbol}")
async def get_price(symbol: str):
    """Fetches real-time price info for a symbol."""
    try:
        return await fetch_price_info(symbol)
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/price/batch")
async def get_batch_prices(symbols: str):
    """Fetches price info for multiple symbols at once."""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results = []
    for sym in symbol_list:
        try:
            results.append(await fetch_price_info(sym))
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return results

@app.get("/indices")
async def get_indices():
    """Fetches key market indices in one call."""
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "GOLD": "GC=F",
        "USD/INR": "USDINR=X"
    }
    results = {}
    for name, sym in indices.items():
        try:
            info = await fetch_price_info(sym)
            results[name] = info
        except:
            results[name] = {"error": "Fetch failed"}
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Helper Logic
# ──────────────────────────────────────────────────────────────────────────────
async def fetch_price_info(symbol: str) -> Dict[str, Any]:
    """Helper to fetch price details using yfinance."""
    tk = yf.Ticker(symbol)
    # Fetch 2 days of data to calculate change
    hist = tk.history(period="2d")
    if hist.empty:
        raise ValueError(f"No price data for {symbol}")
    
    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "change_pct": round(change_pct, 2),
        "prev_close": round(prev_close, 2)
    }

# ──────────────────────────────────────────────────────────────────────────────
# Signal Generation Logic
# ──────────────────────────────────────────────────────────────────────────────
async def generate_signal(symbol: str) -> Dict[str, Any]:
    """
    Logic: RL Model if exists, else Technical Analysis Fallback.
    """
    from data.pipeline import FeatureEngineer, MarketRegimeFeatures
    
    # 1. Fetch recent data (Need ~252 rows for 52w high and EMAs)
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=400)
    
    logger.info(f"Fetching data for {symbol} dynamically...")
    df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df['Symbol'] = symbol
    
    # 2. Engineer Features (TA)
    df = FeatureEngineer.add_price_features(df)
    df = FeatureEngineer.add_technical_indicators(df)
    df = FeatureEngineer.add_volume_features(df)
    df = MarketRegimeFeatures.add_regime_features(df)
    
    latest = df.iloc[-1]
    top_features = {
        "rsi_14": float(latest.get("rsi_14", 50)),
        "macd_line": float(latest.get("macd_line", 0)),
        "volume_ratio": float(latest.get("volume_ratio", 1.0)),
        "price_vs_52w_high": float(latest.get("price_vs_52w_high", 0.5))
    }
    
    model_path = f"models/{symbol}_model"
    
    # Check RL model availability for this SPECIFIC symbol
    if os.path.exists(model_path + ".zip"):
        try:
            # We reuse PaperTrader for RL prediction as it handles agent loading
            trader = PaperTrader(model_path=model_path, symbol=symbol)
            
            # Pad features to 36 for 41-dim observation
            dummy_cols = ["close"] + [f"feat_{i}" for i in range(36)]
            dummy_data = {col: [0.0] for col in dummy_cols}
            dummy_data["close"] = [latest["Close"]]
            
            # Map existing features to dummy cols
            available_feats = [c for c in df.columns if c not in ['Symbol', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            for i, feat in enumerate(available_feats[:36]):
                dummy_data[f"feat_{i}"] = [latest[feat]]
                
            env = TradingEnv(pd.DataFrame(dummy_data))
            trader.load_agent(env)
            
            # Construct observation (last 5 are portfolio placeholders)
            obs = np.concatenate([env._get_observation()[:36], [0, 0, 0, 0, 1]]) # No shares held, neutral state
            
            action_idx, _ = trader.agent.model.predict(obs, deterministic=True)
            action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[int(action_idx)]
            
            return {
                "symbol": symbol,
                "action": action_name,
                "confidence": 0.85, # High confidence for RL
                "method": "RL_MODEL",
                "top_features": top_features
            }
        except Exception as e:
            logger.warning(f"RL fallback failed for {symbol}: {e}. using Technical.")

    # 3. Technical Analysis Fallback
    rsi = top_features["rsi_14"]
    macd = top_features["macd_line"]
    vol = top_features["volume_ratio"]
    
    action = "HOLD"
    confidence = 0.5
    
    if rsi < 35 and vol > 1.2:
        action = "BUY"
        confidence = 0.7
    elif rsi > 65 and vol > 1.2:
        action = "SELL"
        confidence = 0.7
    elif macd > 0 and rsi < 50:
        action = "BUY"
        confidence = 0.6
    elif macd < 0 and rsi > 50:
        action = "SELL"
        confidence = 0.6

    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "method": "TECHNICAL",
        "top_features": top_features
    }

# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
