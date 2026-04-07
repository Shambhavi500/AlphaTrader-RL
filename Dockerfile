# ---------------------------------------------------------------
# AlphaTrader-RL | OpenEnv Submission
# Base: python:3.11-slim  (compatible with Hugging Face Spaces)
# ---------------------------------------------------------------
FROM python:3.11-slim

# Metadata
LABEL maintainer="Shambhavi Patil"
LABEL description="AlphaTrader-RL OpenEnv hackathon submission"
LABEL version="1.0.0"

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements_openenv.txt ./
RUN pip install --no-cache-dir -r requirements_openenv.txt

# Copy source code — core inference files
COPY trading_env.py   ./
COPY graders.py       ./
COPY inference.py     ./
COPY openenv.yaml     ./
COPY .env              ./

# Copy environment module (reward.py lives here)
COPY environment/ ./environment/

# Copy data module (pipeline.py needed for auto-fetch fallback)
RUN mkdir -p data
COPY data/pipeline.py               ./data/
COPY data/processed_market_data.parquet ./data/

# Copy API module (config: API keys, base URL, model name)
COPY api/ ./api/

# Copy LLM module (OpenAI-compatible explainer)
COPY llm/ ./llm/

# Create logs directory
RUN mkdir -p logs

# Health check: verify all imports work
RUN python -c "\
from trading_env import TradingEnv; \
from graders import grade_task1, grade_task2, grade_task3; \
from environment.reward import RewardCalculator; \
from data.pipeline import FeatureEngineer; \
from api import API_BASE_URL, MODEL_NAME; \
from llm.explainer import explain_trade; \
print('All imports OK')"

# Run inference
CMD ["python", "inference.py"]