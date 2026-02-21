# Copilot instructions for ClearSight News

## Big picture
- One Flask app factory in [src/news_insight_app/__init__.py](src/news_insight_app/__init__.py) that registers the `main` blueprint from [src/news_insight_app/main.py](src/news_insight_app/main.py).
- [src/main.py](src/main.py) is the dev entry point only (`python src/main.py`); production uses gunicorn via the `Procfile`.
- API surface: `/api/news`, `/api/news/<id>`, `/api/news/<id>/analysis`, `/api/compare` (POST), `/api/health`, plus `/`, `/news-search`, `/compare` for the UI.
- Mock data lives in `MOCK_NEWS` in [src/news_insight_app/services.py](src/news_insight_app/services.py). API responses enrich each article with:
  - `summary` from `generate_summary()`
  - `sentiment` from `analyze_sentiment()` → `GroqSentimentService.analyze()` (Groq)
  - `insights` from `get_article_insights()`

## AI / Groq layer
- All Groq calls live in [src/news_insight_app/groq_service.py](src/news_insight_app/groq_service.py).
- Three purpose-mapped model constants, each overridable via env var:
  - `GROQ_SENTIMENT_MODEL` (default `llama-3.1-8b-instant`) — used by `GroqSentimentService`
  - `GROQ_RHETORIC_MODEL` (default `llama-3.1-8b-instant`) — used by `analyze_rhetoric()`
  - `GROQ_COMPARISON_MODEL` (default `llama-3.3-70b-versatile`) — used by `compare_article_texts()`
- All Groq calls go through `_chat_completion()` (chat-completions API, not completions). Monkeypatch `_chat_completion` in tests — never mock `requests`.
- `_strip_think()` removes `<think>…</think>` blocks before storing model output.

## UI templates
- Three templates, all in [src/news_insight_app/templates/](src/news_insight_app/templates/):
  - `index.html` — single-article analysis view
  - `news_search.html` — NewsAPI search + article selection
  - `compare.html` — side-by-side rhetoric + comparison view (data loaded client-side from `sessionStorage`)
- Templates consume `article.sentiment` and `article.insights`; keep the response shape stable.
- Sentiment labels are `Positive`/`Negative`/`Neutral` — used as CSS class names.

## Tests
- Pytest configured in [pytest.ini](pytest.ini); tests in [tests/](tests/).
- [tests/conftest.py](tests/conftest.py) provides `app` / `client` fixtures via `create_app()`, and `DummyResponse` for HTTP mocking.
- Keep `generate_summary()`'s ellipsis format (`". .."`) consistent with test expectations.

## Dependencies
- Runtime deps: [requirements.txt](requirements.txt) — Flask, gunicorn, groq, requests, newsapi-python, python-dotenv.
- No TextBlob / NLTK / tokenizer deps — those were removed when local models were replaced with Groq.
- Dev/test deps declared in [setup.py](setup.py) under `extras_require[dev]`.
