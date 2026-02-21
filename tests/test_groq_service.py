"""Tests for groq_service – mocks the Groq SDK client so no real API key is needed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import news_insight_app.groq_service as groq_service_module
from news_insight_app.groq_service import (
    GroqSentimentService,
    analyze_rhetoric,
    compare_article_texts,
    GROQ_SENTIMENT_MODEL,
    GROQ_RHETORIC_MODEL,
    GROQ_COMPARISON_MODEL,
)


# ---------------------------------------------------------------------------
# Fake Groq SDK objects
# ---------------------------------------------------------------------------

def _make_completion(content: str, total_tokens: int = 42) -> Any:
    """Build a minimal object that mimics groq.types.chat.ChatCompletion."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeGroqClient:
    """Fake Groq SDK client that captures calls to chat.completions.create."""

    def __init__(self, content: str = "mocked response", total_tokens: int = 42):
        self._content = content
        self._total_tokens = total_tokens
        self.calls: list[dict] = []

        class _Completions:
            def __init__(inner_self):
                pass

            def create(inner_self, **kwargs):
                self.calls.append(kwargs)
                return _make_completion(self._content, self._total_tokens)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_get_client(monkeypatch):
    """Replace _get_client so tests never need a real GROQ_API_KEY."""
    # Individual tests that need a specific FakeGroqClient can override via
    # monkeypatching _chat_completion directly.
    fake_client = FakeGroqClient()
    monkeypatch.setattr(groq_service_module, "_get_client", lambda: fake_client)


def _patch_chat(monkeypatch, content: str, total_tokens: int = 42):
    """Helper: monkeypatch _chat_completion to return (content, total_tokens)."""
    def fake_chat(model, prompt, max_tokens=500, temperature=0.3, client=None):
        return content, total_tokens
    monkeypatch.setattr(groq_service_module, "_chat_completion", fake_chat)


# ---------------------------------------------------------------------------
# analyze_rhetoric
# ---------------------------------------------------------------------------

def test_analyze_rhetoric_returns_expected_keys(monkeypatch):
    _patch_chat(monkeypatch, "Great rhetorical analysis", 55)
    result = analyze_rhetoric("Some news article text.")
    assert result["analysis"] == "Great rhetorical analysis"
    assert result["text"] == "Great rhetorical analysis"
    assert result["tokens_used"] == 55
    assert result["error"] is None
    assert result["model"] == GROQ_RHETORIC_MODEL


def test_analyze_rhetoric_empty_text():
    result = analyze_rhetoric("")
    assert result["error"] == "No content provided."
    assert result["analysis"] == "Rhetorical analysis unavailable for this story."


def test_analyze_rhetoric_exception_captured(monkeypatch):
    def failing_chat(model, prompt, **kwargs):
        raise RuntimeError("network failure")
    monkeypatch.setattr(groq_service_module, "_chat_completion", failing_chat)
    result = analyze_rhetoric("Some article")
    assert result["error"] is not None
    assert "Groq rhetoric request failed" in result["error"]
    assert result["analysis"] == "Rhetorical analysis unavailable for this story."


# ---------------------------------------------------------------------------
# compare_article_texts
# ---------------------------------------------------------------------------

def test_compare_article_texts_returns_expected_keys(monkeypatch):
    _patch_chat(monkeypatch, "Detailed comparison", 99)
    result = compare_article_texts("Article one text.", "Article two text.")
    assert result["comparison"] == "Detailed comparison"
    assert result["text"] == "Detailed comparison"
    assert result["tokens_used"] == 99
    assert result["error"] is None
    assert result["model"] == GROQ_COMPARISON_MODEL


def test_compare_article_texts_missing_reference():
    result = compare_article_texts("Article one", "")
    assert result["error"] is not None
    assert "empty" in result["error"].lower()
    assert result["comparison"] == "Comparison unavailable for this pair of stories."


def test_compare_article_texts_missing_primary():
    result = compare_article_texts("", "Article two")
    assert result["error"] is not None
    assert "empty" in result["error"].lower()


def test_compare_article_texts_exception_captured(monkeypatch):
    def failing_chat(model, prompt, **kwargs):
        raise RuntimeError("timeout")
    monkeypatch.setattr(groq_service_module, "_chat_completion", failing_chat)
    result = compare_article_texts("Art 1", "Art 2")
    assert result["error"] is not None
    assert "Groq comparison request failed" in result["error"]


# ---------------------------------------------------------------------------
# GroqSentimentService
# ---------------------------------------------------------------------------

def test_groq_sentiment_empty_text_short_circuits(monkeypatch):
    called = []

    def fake_chat(model, prompt, **kwargs):
        called.append(True)
        return "", 0

    monkeypatch.setattr(groq_service_module, "_chat_completion", fake_chat)
    service = GroqSentimentService()
    result = service.analyze("")

    assert called == []
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


def test_groq_sentiment_positive(monkeypatch):
    _patch_chat(
        monkeypatch,
        '{"sentiment": "positive", "tone": "calm", "evidence": ["great result"]}',
    )
    service = GroqSentimentService()
    result = service.analyze("This is great news.")
    assert result["sentiment"] == "Positive"
    assert result["polarity"] == pytest.approx(1.0)
    assert result["label"] == "POSITIVE"
    assert result["score"] == pytest.approx(1.0)
    assert result["model"] == GROQ_SENTIMENT_MODEL


def test_groq_sentiment_negative(monkeypatch):
    _patch_chat(
        monkeypatch,
        '{"sentiment": "negative", "tone": "emotional", "evidence": ["terrible"]}',
    )
    service = GroqSentimentService()
    result = service.analyze("This is terrible news.")
    assert result["sentiment"] == "Negative"
    assert result["polarity"] == pytest.approx(-1.0)
    assert result["label"] == "NEGATIVE"


def test_groq_sentiment_neutral(monkeypatch):
    _patch_chat(monkeypatch, '{"sentiment": "neutral", "tone": "calm", "evidence": []}')
    service = GroqSentimentService()
    result = service.analyze("Things happened today.")
    assert result["sentiment"] == "Neutral"
    assert result["polarity"] == pytest.approx(0.0)
    assert result["label"] == "NEUTRAL"
    assert result["score"] == pytest.approx(0.0)


def test_groq_sentiment_fallback_keyword_detection(monkeypatch):
    """When the model returns plain text instead of JSON, keywords drive the label."""
    _patch_chat(monkeypatch, "The sentiment is clearly negative overall.")
    service = GroqSentimentService()
    result = service.analyze("Some article text.")
    assert result["sentiment"] == "Negative"


def test_groq_sentiment_custom_model(monkeypatch):
    _patch_chat(monkeypatch, '{"sentiment": "positive", "tone": "calm", "evidence": []}')
    service = GroqSentimentService(model_name="my-custom-model")
    result = service.analyze("Good news.")
    assert result["model"] == "my-custom-model"


def test_groq_sentiment_exception_returns_neutral(monkeypatch):
    def failing_chat(model, prompt, **kwargs):
        raise RuntimeError("API down")
    monkeypatch.setattr(groq_service_module, "_chat_completion", failing_chat)
    service = GroqSentimentService()
    result = service.analyze("Some text.")
    # Should degrade gracefully; raw_text will be "" → neutral fallback
    assert result["sentiment"] == "Neutral"
    assert isinstance(result["latency_ms"], int)


# ---------------------------------------------------------------------------
# _extract_first_json  (utility)
# ---------------------------------------------------------------------------

def test_extract_first_json_valid():
    from news_insight_app.groq_service import _extract_first_json
    obj = _extract_first_json('prefix {"key": "value"} suffix')
    assert obj == {"key": "value"}


def test_extract_first_json_no_json():
    from news_insight_app.groq_service import _extract_first_json
    assert _extract_first_json("no json here") is None


# ---------------------------------------------------------------------------
# _strip_think
# ---------------------------------------------------------------------------

def test_strip_think_removes_block():
    from news_insight_app.groq_service import _strip_think
    raw = "<think>\nLet me reason about this...\n</think>\nActual answer here."
    assert _strip_think(raw) == "Actual answer here."


def test_strip_think_multiline():
    from news_insight_app.groq_service import _strip_think
    raw = "<think>\nline 1\nline 2\n</think>\n\nAnswer."
    assert _strip_think(raw) == "Answer."


def test_strip_think_no_block_unchanged():
    from news_insight_app.groq_service import _strip_think
    raw = "Plain response with no think block."
    assert _strip_think(raw) == raw


def test_strip_think_case_insensitive():
    from news_insight_app.groq_service import _strip_think
    raw = "<THINK>hidden</THINK>visible"
    assert _strip_think(raw) == "visible"


def test_analyze_rhetoric_strips_think(monkeypatch):
    _patch_chat(monkeypatch, "<think>\ninternal reasoning\n</think>\nRhetoric result.")
    result = analyze_rhetoric("Article text.")
    assert result["analysis"] == "Rhetoric result."
    assert "<think>" not in result["analysis"]


def test_compare_article_texts_strips_think(monkeypatch):
    _patch_chat(monkeypatch, "<think>reasoning</think>Comparison result.")
    result = compare_article_texts("Article one.", "Article two.")
    assert result["comparison"] == "Comparison result."
    assert "<think>" not in result["comparison"]


# ---------------------------------------------------------------------------
# _chat_completion uses messages format (not completions/prompt)
# ---------------------------------------------------------------------------

def test_chat_completion_sends_messages_format(monkeypatch):
    """Verify _chat_completion passes a messages list to the Groq client."""
    fake_client = FakeGroqClient(content="hello", total_tokens=7)

    # Bypass _get_client by passing the fake client directly
    text, tokens = groq_service_module._chat_completion(
        model="test-model",
        prompt="test prompt",
        max_tokens=100,
        temperature=0.5,
        client=fake_client,
    )

    assert text == "hello"
    assert tokens == 7
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "test prompt"}]
    assert call["max_tokens"] == 100
    assert call["temperature"] == pytest.approx(0.5)
