"""Groq-backed analysis and sentiment services.

Uses the ``groq`` Python SDK (chat-completions API) with three purpose-mapped
models, each overridable via environment variable:

Model mapping
-------------
* Sentiment / tone   → GROQ_SENTIMENT_MODEL  (default: ``llama-3.1-8b-instant``)
* Rhetoric analysis  → GROQ_RHETORIC_MODEL   (default: ``llama-3.1-8b-instant``)
* Comparison         → GROQ_COMPARISON_MODEL (default: ``llama-3.3-70b-versatile``)

All three model slugs are overridable via environment variables so the exact
Groq model can be adjusted without code changes.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from groq import Groq

# ---------------------------------------------------------------------------
# Model name configuration
# ---------------------------------------------------------------------------

# Fast, cheap production model used for both sentiment and rhetoric.
# Swap to a heavier model via env var, e.g. GROQ_RHETORIC_MODEL=llama-3.3-70b-versatile
GROQ_SENTIMENT_MODEL: str = os.getenv("GROQ_SENTIMENT_MODEL", "llama-3.1-8b-instant")
GROQ_RHETORIC_MODEL: str = os.getenv("GROQ_RHETORIC_MODEL", "llama-3.1-8b-instant")
GROQ_COMPARISON_MODEL: str = os.getenv("GROQ_COMPARISON_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    """Return a Groq SDK client using GROQ_API_KEY from the environment."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _truncate_text(text: str, limit: int = 4000) -> str:
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[:limit].rstrip() + " ..."


def _build_response(model_name: str, default_text: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "tokens_used": 0,
        "error": None,
        "text": default_text,
    }


def _chat_completion(
    model: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    client: Optional[Groq] = None,
) -> tuple[str, int]:
    """Send a single-turn chat completion to Groq and return (text, total_tokens).

    Separating this into its own function makes it easy to monkeypatch in tests.
    """
    groq_client = client or _get_client()
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = (completion.choices[0].message.content or "").strip()
    tokens: int = completion.usage.total_tokens if completion.usage else 0
    return text, tokens


def _extract_first_json(text: str) -> Any:
    """Return the first valid JSON object found in *text*, or ``None``."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by Qwen3 models.

    The block may span multiple lines. Everything after the closing tag (if
    present) is returned trimmed; if there is no closing tag the entire text
    is returned unchanged so we never silently lose content.
    """
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Public analysis functions  (same interface as analysis_service.py)
# ---------------------------------------------------------------------------

def analyze_rhetoric(article_text: str) -> Dict[str, Any]:
    """Analyse *article_text* for tone and rhetorical devices via Groq.

    Returns a dict with keys: ``model``, ``tokens_used``, ``error``, ``text``,
    and ``analysis``.
    """
    result = _build_response(
        GROQ_RHETORIC_MODEL,
        "Rhetorical analysis unavailable for this story.",
    )
    result["analysis"] = result["text"]

    trimmed = _truncate_text(article_text)
    if not trimmed:
        result["error"] = "No content provided."
        return result

    prompt = f"""Analyze this news article for tone and rhetorical devices.

Article:
{trimmed}

Provide analysis in this format:
1. Overall Tone: (e.g., neutral, persuasive, alarmist, celebratory)
2. Sentiment: (positive, negative, or neutral with confidence score)
3. Rhetorical Devices Found:
   - List specific devices used (metaphors, appeals to emotion, repetition, loaded language, etc.)
   - Quote examples from the text
4. Bias Indicators: Any signs of bias or framing

Analysis:"""

    try:
        analysis_text, tokens = _chat_completion(
            GROQ_RHETORIC_MODEL, prompt, max_tokens=500
        )
        analysis_text = _strip_think(analysis_text)
        result.update(
            {
                "analysis": analysis_text or result["text"],
                "tokens_used": tokens,
            }
        )
        result["text"] = result["analysis"]
    except Exception as exc:
        result["error"] = f"Groq rhetoric request failed: {exc}"

    return result


def compare_article_texts(primary_text: str, reference_text: str) -> Dict[str, Any]:
    """Compare two articles for framing, tone, and bias via Groq.

    Returns a dict with keys: ``model``, ``tokens_used``, ``error``, ``text``,
    and ``comparison``.
    """
    result = _build_response(
        GROQ_COMPARISON_MODEL,
        "Comparison unavailable for this pair of stories.",
    )
    result["comparison"] = result["text"]

    primary = _truncate_text(primary_text)
    reference = _truncate_text(reference_text)

    if not primary or not reference:
        result["error"] = "One of the articles was empty."
        return result

    prompt = f"""Compare these two news articles covering similar topics.

Article 1:
{primary}

Article 2:
{reference}

Provide comparison in this format:
1. Framing Differences: How does each article frame the story?
2. Tone Comparison: Compare the tone and emotional appeal
3. Source Selection: Note any differences in sources cited or perspectives included
4. Key Differences: What facts or angles does one include that the other doesn't?
5. Bias Assessment: Which article appears more balanced?

Comparison:"""

    try:
        comparison_text, tokens = _chat_completion(
            GROQ_COMPARISON_MODEL, prompt, max_tokens=600
        )
        comparison_text = _strip_think(comparison_text)
        result.update(
            {
                "comparison": comparison_text or result["text"],
                "tokens_used": tokens,
            }
        )
        result["text"] = result["comparison"]
    except Exception as exc:
        result["error"] = f"Groq comparison request failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# GroqSentimentService  (same interface as SentimentService in sentiment_service.py)
# ---------------------------------------------------------------------------

class GroqSentimentService:
    """Sentiment and tone classifier backed by Groq.

    Defaults to ``llama-3.1-8b-instant`` via ``GROQ_SENTIMENT_MODEL``.
    Exposes an ``analyze(text)`` method that returns a sentiment dict.
    """

    def __init__(self, model_name: str = GROQ_SENTIMENT_MODEL) -> None:
        self.model_name = model_name

    def analyze(self, text: str) -> Dict[str, Any]:
        """Return a sentiment dict for *text*.

        Returns:
            A dict with keys: ``sentiment``, ``polarity``, ``subjectivity``,
            ``model``, ``confidence``, ``label``, ``score``, ``raw``,
            ``token_count``, ``latency_ms``.
        """
        if not text:
            return {
                "sentiment": "Neutral",
                "polarity": 0.0,
                "subjectivity": 0.0,
                "model": self.model_name,
                "confidence": 0.0,
                "label": "NEUTRAL",
                "score": 0.0,
                "raw": None,
                "token_count": 0,
                "latency_ms": 0,
            }

        prompt = (
            "You are a sentiment and tone classifier. Return JSON only with no other text.\n"
            "Article:\n"
            f"{text}\n\n"
            "Classify:\n"
            "- sentiment: positive, negative, or neutral\n"
            "- tone: calm | emotional | inflammatory | persuasive | neutral | sarcastic | urgent\n"
            "- evidence: list of phrases that influenced your classification\n"
        )

        start_time = time.perf_counter()
        raw_text = ""
        try:
            raw_text, _ = _chat_completion(self.model_name, prompt, max_tokens=200)
        except Exception:
            pass
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Parse JSON response -------------------------------------------------
        raw_parsed: Any = raw_text
        parsed = _extract_first_json(raw_text)
        if parsed is not None:
            raw_parsed = parsed
            sentiment_raw = str(parsed.get("sentiment", "neutral")).lower()
        else:
            lower = raw_text.lower()
            if "negative" in lower:
                sentiment_raw = "negative"
            elif "positive" in lower:
                sentiment_raw = "positive"
            else:
                sentiment_raw = "neutral"

        tone: str = ""
        evidence: List[Any] = []
        if isinstance(parsed, dict):
            tone = str(parsed.get("tone", ""))
            evidence = parsed.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []

        if sentiment_raw == "negative":
            label, sentiment, polarity = "NEGATIVE", "Negative", -1.0
        elif sentiment_raw == "positive":
            label, sentiment, polarity = "POSITIVE", "Positive", 1.0
        else:
            label, sentiment, polarity = "NEUTRAL", "Neutral", 0.0

        score = 1.0 if sentiment_raw != "neutral" else 0.0

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": 1.0,
            "model": self.model_name,
            "confidence": score,
            "label": label,
            "score": score,
            "raw": raw_parsed,
            "token_count": 0,  # Groq counts tokens server-side; not re-counted locally
            "latency_ms": latency_ms,
        }
