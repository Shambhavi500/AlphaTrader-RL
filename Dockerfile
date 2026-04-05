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

# Copy source code and pre-processed data
COPY trading_env.py   ./
COPY graders.py       ./
COPY inference.py     ./
COPY openenv.yaml     ./

# Copy only what's needed from data/ (the bundled parquet — no raw files)
RUN mkdir -p data
COPY data/processed_market_data.parquet data/

# Create logs directory
RUN mkdir -p logs

# Health check: verify imports work
RUN python -c "from trading_env import TradingEnv; from graders import grade_task1; print('Import OK')"

# Run inference
CMD ["python", "inference.py"]
