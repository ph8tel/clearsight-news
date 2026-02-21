import pytest

import news_insight_app.sentiment_service as sentiment_service_module
from news_insight_app.sentiment_service import SentimentService, PHI_MODEL_NAME


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_sentiment_service_positive_label(monkeypatch):
    def fake_post(url, json, timeout):
        return _DummyResponse({
            "choices": [{"text": '{"sentiment": "positive", "tone": "calm", "evidence": ["love"]}'}],
            "usage": {"total_tokens": 15},
        })

    monkeypatch.setattr(sentiment_service_module.requests, "post", fake_post)
    service = SentimentService()

    result = service.analyze("I love this product.")

    assert result["sentiment"] == "Positive"
    assert result["polarity"] == pytest.approx(1.0, rel=1e-3)
    assert result["subjectivity"] == pytest.approx(1.0)
    assert result["model"] == PHI_MODEL_NAME
    assert result["label"] == "POSITIVE"
    assert result["score"] == pytest.approx(1.0, rel=1e-3)
    assert result["token_count"] > 0
    assert isinstance(result["latency_ms"], int)


def test_sentiment_service_negative_label(monkeypatch):
    def fake_post(url, json, timeout):
        return _DummyResponse({
            "choices": [{"text": '{"sentiment": "negative", "tone": "emotional", "evidence": ["hate"]}'}],
            "usage": {"total_tokens": 15},
        })

    monkeypatch.setattr(sentiment_service_module.requests, "post", fake_post)
    service = SentimentService()

    result = service.analyze("I hate this product.")

    assert result["sentiment"] == "Negative"
    assert result["polarity"] == pytest.approx(-1.0, rel=1e-3)
    assert result["subjectivity"] == pytest.approx(1.0)
    assert result["label"] == "NEGATIVE"
    assert result["score"] == pytest.approx(1.0, rel=1e-3)
    assert result["token_count"] > 0
    assert isinstance(result["latency_ms"], int)


def test_sentiment_service_empty_text_short_circuits(monkeypatch):
    called = []

    def fake_post(url, json, timeout):
        called.append(True)
        return _DummyResponse({"choices": [], "usage": {}})

    monkeypatch.setattr(sentiment_service_module.requests, "post", fake_post)
    service = SentimentService()

    result = service.analyze("")

    assert result == {
        "sentiment": "Neutral",
        "polarity": 0.0,
        "subjectivity": 0.0,
        "model": service.model_name,
        "confidence": 0.0,
        "label": "NEUTRAL",
        "score": 0.0,
        "raw": None,
        "token_count": 0,
        "latency_ms": 0,
    }
    assert called == []
