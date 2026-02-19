import pytest

import news_insight_app.services as services


class DummySentimentService:
    def __init__(self):
        self.calls = []

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


def test_chunk_text_splits_long_text():
    text = ". ".join(["Sentence"] * 30) + "."
    chunks = services._chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)


def test_analyze_sentiment_uses_single_chunk(monkeypatch):
    dummy = DummySentimentService()
    monkeypatch.setattr(services, "_get_sentiment_service", lambda: dummy)

    text = "Short text."
    result = services.analyze_sentiment(text)

    assert dummy.calls == ["Short text."]
    assert result["sentiment"] == "Neutral"


def test_analyze_sentiment_uses_first_chunk_for_long_text(monkeypatch):
    dummy = DummySentimentService()
    monkeypatch.setattr(services, "_get_sentiment_service", lambda: dummy)

    text = ". ".join(["Sentence"] * 40) + "."
    result = services.analyze_sentiment(text)

    assert len(dummy.calls) == 1
    assert dummy.calls[0].endswith(".")
    assert result["model"] == "dummy"


def test_get_article_insights_returns_expected_fields():
    text = "Alpha beta. Gamma delta."
    insights = services.get_article_insights(text)

    assert insights["word_count"] == len(text.split())
    assert insights["sentence_count"] == 2
    assert isinstance(insights["keywords"], list)
    assert isinstance(insights["reading_time_minutes"], int)
    assert insights["reading_time_minutes"] > 0 # Test that the reading time is positive.

