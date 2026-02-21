# ClearSight News

A sandbox app for side-by-side political news analysis powered by Groq-hosted Llama models. Compare how different outlets frame the same story — sentiment, rhetoric, and framing differences surfaced in one view.

Deployed on Heroku. Built with Flask + Groq.

---

## Features

- **Sentiment & tone** — classifies each article as Positive / Negative / Neutral with tone labels (calm, emotional, inflammatory, etc.) via `llama-3.1-8b-instant`
- **Rhetorical analysis** — breaks down tone, rhetorical devices, loaded language, and bias indicators via `llama-3.1-8b-instant`
- **Cross-article comparison** — frames differences, source selection, and bias assessment between two articles via `llama-3.3-70b-versatile`
- **Live news search** — fetches articles from left/right source buckets via NewsAPI and feeds them into the compare view

---

## Project layout

```
src/
  main.py                        # Flask entry point (dev)
  news_insight_app/
    __init__.py                  # App factory (used by gunicorn)
    main.py                      # Blueprint with all routes
    groq_service.py              # Groq API calls (rhetoric, comparison, sentiment)
    services.py                  # Article helpers (summary, keywords, insights)
    news_api_service.py          # NewsAPI wrapper
    templates/
      index.html                 # Single-article analysis view
      compare.html               # Side-by-side comparison view
      news_search.html           # Search & select view
tests/                           # pytest suite
Procfile                         # gunicorn entry point for Heroku
runtime.txt                      # Python version pin
requirements.txt
```

---

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env_example` to `.env` and fill in your keys:

```
NEWS_API_KEY=your_newsapi_key
GROQ_API_KEY=your_groq_api_key
```

Run the dev server:

```bash
python src/main.py
```

Visit `http://localhost:5000`.

---

## Deploying to Heroku

```bash
heroku create clearsight-news
heroku config:set GROQ_API_KEY=... NEWS_API_KEY=...
git push heroku main
```

The `Procfile` starts gunicorn with 2 workers / 4 threads:

```
web: gunicorn "news_insight_app:create_app()" --workers=2 --threads=4 --timeout=120 --bind=0.0.0.0:$PORT
```

---

## Models

All three model slugs are overridable via environment variables without code changes:

| Purpose | Default model | Env var |
|---|---|---|
| Sentiment / tone | `llama-3.1-8b-instant` | `GROQ_SENTIMENT_MODEL` |
| Rhetorical analysis | `llama-3.1-8b-instant` | `GROQ_RHETORIC_MODEL` |
| Cross-article comparison | `llama-3.3-70b-versatile` | `GROQ_COMPARISON_MODEL` |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Single-article analysis UI |
| `/news-search` | GET | Search & compare UI |
| `/compare` | GET | Side-by-side comparison view |
| `/api/news` | GET | All mock articles with sentiment + insights |
| `/api/news/<id>` | GET | Single article |
| `/api/news/<id>/analysis` | GET | Deep rhetoric + comparison for one article |
| `/api/compare` | POST | Rhetoric + comparison for two supplied articles |
| `/api/health` | GET | Health check |

---

## Tests

```bash
python -m pytest
```

Configured in `pytest.ini`. All external API calls are monkeypatched — no real keys needed to run the suite.
