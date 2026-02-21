import pytest

import news_insight_app.services as services


class DummySentimentService:
    def __init__(self):
        self.calls = []
        self.model_name = "dummy"

    def analyze(self, text):
        self.calls.append(text)
        return {
            "sentiment": "Neutral",
            "polarity": 0.0,
            "subjectivity": 1.0,
            "model": "dummy",
            "confidence": 0.0,
            "label": "NEUTRAL",
            "score": 0.0,
            "raw": {"label": "NEUTRAL", "score": 0.0},
            "token_count": len(text.split()),
            "latency_ms": 0,
        }


def test_generate_summary_ellipsis():
    text = "One. Two. Three."
    summary = services.generate_summary(text, max_sentences=2)
    assert summary.endswith(". ..")


def test_analyze_sentiment_uses_single_chunk(monkeypatch):
    dummy = DummySentimentService()
    monkeypatch.setattr(services, "_get_sentiment_service", lambda: dummy)

    text = "Short text."
    result = services.analyze_sentiment(text)

    assert dummy.calls == ["Short text."]
    assert result["sentiment"] == "Neutral"


def test_analyze_sentiment_passes_full_text(monkeypatch):
    """With a 131K context window there is no chunking — full text goes straight to the model."""
    dummy = DummySentimentService()
    monkeypatch.setattr(services, "_get_sentiment_service", lambda: dummy)

    text = " ".join(["word"] * 500)  # 500-word article, no chunking
    services.analyze_sentiment(text)

    assert len(dummy.calls) == 1
    assert dummy.calls[0] == text


def test_analyze_sentiment_empty_text(monkeypatch):
    dummy = DummySentimentService()
    monkeypatch.setattr(services, "_get_sentiment_service", lambda: dummy)

    services.analyze_sentiment("")
    assert dummy.calls == [""]


def test_get_article_insights_returns_expected_fields():
    text = "Alpha beta. Gamma delta."
    insights = services.get_article_insights(text)

    assert insights["word_count"] == len(text.split())
    assert insights["sentence_count"] == 2
    assert isinstance(insights["keywords"], list)
    assert isinstance(insights["reading_time_minutes"], int)
    assert insights["reading_time_minutes"] > 0

