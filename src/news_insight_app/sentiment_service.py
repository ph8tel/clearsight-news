from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict

import requests

from .tokenizer_utils import get_tokenizer_provider

PHI_URL = os.getenv("PHI_ANALYSIS_URL", "http://192.168.1.108:8002/v1/completions")
PHI_MODEL_NAME = "phi3.5:latest"


class SentimentService:
    def __init__(
        self,
        model_name: str = PHI_MODEL_NAME,
        phi_url: str = PHI_URL,
    ) -> None:
        self.model_name = model_name
        self._phi_url = phi_url

    def analyze(self, text: str) -> Dict[str, Any]:
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
        body: Dict[str, Any] = {}
        try:
            response = requests.post(
                self._phi_url,
                json={"prompt": prompt, "max_tokens": 200, "temperature": 0.3},
                timeout=60,
            )
            response.raise_for_status()
            body = response.json()
        except Exception:
            pass
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        choices = body.get("choices") or []
        raw_text = choices[0].get("text", "").strip() if choices else ""

        sentiment_raw = "neutral"
        raw_parsed: Any = raw_text
        try:
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                raw_parsed = json.loads(json_match.group())
                sentiment_raw = str(raw_parsed.get("sentiment", "neutral")).lower()
        except (json.JSONDecodeError, AttributeError):
            lower = raw_text.lower()
            if "negative" in lower:
                sentiment_raw = "negative"
            elif "positive" in lower:
                sentiment_raw = "positive"

        if sentiment_raw == "negative":
            label, sentiment, polarity = "NEGATIVE", "Negative", -1.0
        elif sentiment_raw == "positive":
            label, sentiment, polarity = "POSITIVE", "Positive", 1.0
        else:
            label, sentiment, polarity = "NEUTRAL", "Neutral", 0.0

        score = 1.0 if sentiment_raw != "neutral" else 0.0

        provider = get_tokenizer_provider()
        token_count = provider.count_tokens(text, self.model_name)

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": 1.0,
            "model": self.model_name,
            "confidence": score,
            "label": label,
            "score": score,
            "raw": raw_parsed,
            "token_count": token_count,
            "latency_ms": latency_ms,
        }