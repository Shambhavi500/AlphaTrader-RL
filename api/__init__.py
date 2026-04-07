# AlphaTrader-RL API Package
# ──────────────────────────────────────────────────────────────────────────────
# Central configuration: API keys, base URLs, model names.
# All secrets are loaded from .env (never hardcoded).
# ──────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Groq API — OpenAI-compatible endpoint
API_KEY = os.getenv("GROQ_API_KEY", "")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

# HuggingFace token (used for Space deployment / private model access)
HF_TOKEN = os.getenv("HF_TOKEN", "")
