"""
AlphaTrader-RL | LLM Explainer (OpenAI-compatible — xAI Grok)
==============================================================
Uses the OpenAI Python SDK pointed at xAI's API (https://api.x.ai/v1)
to generate plain-English explanations of trading decisions.

Functions
---------
explain_trade(action, features, portfolio_info)
    → 2-3 sentence explanation of a single trading decision.

explain_backtest_summary(metrics_dict)
    → paragraph summarising overall agent performance vs Buy-and-Hold.

explain_live_signal(symbol, signal, sentiment, news_summary)
    → concise rationale for a live trading signal based on sentiment.
"""
import logging
from typing import Any, Dict

from openai import OpenAI

from api import API_KEY, API_BASE_URL, MODEL_NAME

logger = logging.getLogger("LLMExplainer")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client (xAI Grok)
# ──────────────────────────────────────────────────────────────────────────────
_client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

_SYSTEM_PROMPT = (
    "You are a professional financial analyst for AlphaTrader-RL. "
    "Provide clear, jargon-free explanations to investors."
)


def _call_llm(prompt: str) -> str:
    """
    Send a chat-completion request via the OpenAI SDK.
    Falls back gracefully on any error.
    """
    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[AI Unavailable] {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def explain_trade(
    action: str,
    features: Dict[str, Any],
    portfolio_info: Dict[str, Any],
) -> str:
    """
    Generate a 2-3 sentence explanation of a single trading decision.
    """
    feature_lines = "\n".join(
        f"  • {k}: {v:.4f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in features.items()
    )
    portfolio_lines = "\n".join(
        f"  • {k}: {v:.2f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in portfolio_info.items()
    )

    prompt = f"""The trading agent just decided to **{action}**.

Market context:
{feature_lines}

Portfolio state:
{portfolio_lines}

Explain in 2-3 clear sentences why the agent likely made this {action} decision. Focus on the relationship between the feature values and the action. Do not use bullets."""

    logger.info("explain_trade called: action=%s", action)
    return _call_llm(prompt)


def explain_backtest_summary(metrics_dict: Dict[str, Any]) -> str:
    """
    Generate a paragraph summarising the agent's overall backtest performance.
    """
    metrics_lines = "\n".join(
        f"  • {k}: {v:.3f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in metrics_dict.items()
    )

    alpha = metrics_dict.get("alpha_pct", 0.0)
    beat_bah = "outperformed" if alpha >= 0 else "underperformed"

    prompt = f"""Summarise the following backtest results for an RL trading agent:
{metrics_lines}

The agent {beat_bah} the benchmark by {abs(alpha):.2f}%.

Write one coherent paragraph (4-6 sentences) explaining the return vs risk profile, comparing it to Buy-and-Hold, and giving a final verdict on performance."""

    logger.info("explain_backtest_summary called: alpha_pct=%s", alpha)
    return _call_llm(prompt)


def explain_live_signal(
    symbol: str,
    signal: str,
    sentiment: float,
    news_summary: str,
) -> str:
    """
    Provide rationale for a live signal based on market sentiment.
    """
    prompt = f"""Provide a concise (1-2 sentence) rationale for a live trading signal.

Symbol: {symbol}
Signal: {signal}
Sentiment Score: {sentiment:.2f} (-1 to +1)
Recent News: {news_summary}

Why is this signal being generated now?"""

    logger.info("explain_live_signal called: symbol=%s, signal=%s", symbol, signal)
    return _call_llm(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# Self-Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing xAI Grok integration — model: {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}\n")

    test_features = {"rsi_14": 28.5, "volume_ratio": 2.1}
    test_portfolio = {"total_return_pct": 5.2, "shares_held": 0}

    print("--- Trade Explanation Test ---")
    print(explain_trade("BUY", test_features, test_portfolio))

    print("\n--- Live Signal Test ---")
    print(explain_live_signal(
        "TATASTEEL.NS", "BUY", 0.75,
        "Strong quarterly earnings reported today."
    ))
