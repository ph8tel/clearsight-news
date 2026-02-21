from flask import Blueprint, render_template, jsonify, request
from datetime import datetime

from .groq_service import analyze_rhetoric, compare_article_texts
from .services import (
	MOCK_NEWS,
	generate_summary,
	analyze_sentiment,
	get_article_insights,
)
from .news_api_service import NewsApiService

main = Blueprint('main', __name__)


def _serialize_article(article):
    summary = generate_summary(article['content'])
    sentiment = analyze_sentiment(article['content'])
    insights = get_article_insights(article['content'])
    return {
        "id": article['id'],
        "title": article['title'],
        "summary": summary,
        "content": article['content'],
        "url": article['url'],
        "source": article['source'],
        "published_at": article['published_at'],
        "sentiment": sentiment,
        "insights": insights,
    }


def _find_article(article_id):
    return next((a for a in MOCK_NEWS if a['id'] == article_id), None)

@main.route('/')
def index():
    """Landing page — search and compare."""
    return _news_search_view()


@main.route('/news-search')
def news_search_redirect():
    """Legacy URL — redirect to /."""
    from flask import redirect
    qs = request.query_string.decode()
    return redirect(f'/?{qs}' if qs else '/', code=301)


def _process_api_article(article):
    """Normalise a raw NewsAPI article dict into a template-ready dict."""
    content_text = (
        article.get('content')
        or article.get('description')
        or article.get('title')
        or ''
    )
    sentiment = analyze_sentiment(content_text)
    # Promote Phi-specific fields before stripping raw
    raw = sentiment.get('raw') or {}
    tone = raw.get('tone', '') if isinstance(raw, dict) else ''
    evidence = raw.get('evidence', []) if isinstance(raw, dict) else []
    # Strip the raw field so the dict stays JSON-serialisable
    sentiment = {k: v for k, v in sentiment.items() if k != 'raw'}
    sentiment['tone'] = tone
    sentiment['evidence'] = evidence if isinstance(evidence, list) else []
    return {
        'title': article.get('title', 'Untitled'),
        'url': article.get('url', '#'),
        'source': article.get('source', 'Unknown'),
        'published_at': article.get('published_at', '') or article.get('publishedAt', ''),
        'summary': generate_summary(content_text),
        'sentiment': sentiment,
        'insights': get_article_insights(content_text),
        'content': content_text,
        'description': article.get('description', ''),
    }


def _fetch_side(service, query, side, max_articles=5):
    """Fetch and process articles for one political-lean bucket."""
    try:
        raw = service.search_news(query, max_articles=max_articles, source_category=side)
        return [_process_api_article(a) for a in raw], None
    except Exception as exc:
        return [], str(exc)


def _news_search_view():
    """Shared logic for the search + compare landing page."""
    query = request.args.get('q', '').strip()
    left_articles = []
    right_articles = []
    error = None

    if query:
        try:
            service = NewsApiService()
            left_articles, left_err = _fetch_side(service, query, 'left')
            right_articles, right_err = _fetch_side(service, query, 'right')
            error = left_err or right_err
        except Exception as exc:
            error = str(exc)

    return render_template(
        'news_search.html',
        query=query,
        left_articles=left_articles,
        right_articles=right_articles,
        error=error,
    )

@main.route('/api/news')
def get_news():
    """API endpoint to get all news articles"""

    return jsonify([_serialize_article(article) for article in MOCK_NEWS])

@main.route('/api/news/<int:article_id>')
def get_article(article_id):
    """API endpoint to get a specific article"""
    article = _find_article(article_id)
    if not article:
        return jsonify({"error": "Article not found"}), 404
    return jsonify(_serialize_article(article))


@main.route('/api/news/<int:article_id>/analysis')
def get_article_analysis(article_id):
    """Deep analysis (rhetoric + comparison) for a specific article"""
    article = _find_article(article_id)
    if not article:
        return jsonify({"error": "Article not found"}), 404

    article_payload = _serialize_article(article)
    reference_article = next((a for a in MOCK_NEWS if a['id'] != article_id), None)

    if reference_article:
        comparison = compare_article_texts(article['content'], reference_article['content'])
        comparison["reference"] = {
            "id": reference_article['id'],
            "title": reference_article['title'],
        }
    else:
        from .groq_service import GROQ_COMPARISON_MODEL
        comparison = {
            "comparison": "Comparison unavailable; only one article configured.",
            "model": GROQ_COMPARISON_MODEL,
            "tokens_used": 0,
            "error": "No reference article available.",
            "reference": None,
        }

    rhetoric = analyze_rhetoric(article['content'])

    return jsonify({
        "article": article_payload,
        "rhetoric": rhetoric,
        "comparison": comparison,
    })

@main.route('/compare')
def compare():
    """Comparison view shell — article data loaded client-side from sessionStorage."""
    return render_template('compare.html')


@main.route('/api/compare', methods=['POST'])
def compare_articles_api():
    """Run rhetoric + comparison analysis on two externally supplied articles."""
    data = request.get_json(force=True) or {}
    primary = data.get('primary', {})
    reference = data.get('reference', {})

    primary_content = primary.get('content', '')
    reference_content = reference.get('content', '')

    if not primary_content or not reference_content:
        return jsonify({'error': 'Both articles must have content.'}), 400

    primary_rhetoric = analyze_rhetoric(primary_content)
    reference_rhetoric = analyze_rhetoric(reference_content)
    comparison = compare_article_texts(primary_content, reference_content)
    comparison['reference'] = {
        'title': reference.get('title', ''),
        'source': reference.get('source', ''),
    }

    return jsonify({
        'primary': {'meta': primary, 'rhetoric': primary_rhetoric},
        'reference': {'meta': reference, 'rhetoric': reference_rhetoric},
        'comparison': comparison,
    })


@main.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})