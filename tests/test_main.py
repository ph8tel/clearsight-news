import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, generate_summary, analyze_sentiment, extract_keywords, get_article_insights

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test the main index route"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Both Eyes Open' in response.data

def test_health_check_route(client):
    """Test the health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_get_news_route(client):
    """Test the get news endpoint"""
    response = client.get('/api/news')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert 'title' in data[0]
    assert 'summary' in data[0]
    assert 'sentiment' in data[0]
    assert 'insights' in data[0]

def test_generate_summary():
    """Test the summary generation function"""
    from main import generate_summary
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    summary = generate_summary(text, max_sentences=2)
    assert summary.count('.') == 4  # Should have 2 sentences and ..
    assert summary.endswith('. ..')  # Should end with the ellipsis
def test_get_article_route(client):
    """Test the get specific article endpoint"""
    response = client.get('/api/news/1')
    assert response.status_code == 200
    data = response.get_json()
    assert 'title' in data
    assert data['id'] == 1
    assert 'summary' in data
    assert 'sentiment' in data
    assert 'insights' in data

def test_get_nonexistent_article_route(client):
    """Test getting a non-existent article"""
    response = client.get('/api/news/999')
    assert response.status_code == 404


def test_analyze_sentiment():
    """Test the sentiment analysis function"""
    from main import analyze_sentiment
    positive_text = "I love this product. It's amazing!"
    negative_text = "I hate this product. It's terrible!"
    neutral_text = "The weather is nice today."
    
    pos_sentiment = analyze_sentiment(positive_text)
    neg_sentiment = analyze_sentiment(negative_text)
    neutral_sentiment = analyze_sentiment(neutral_text)
    
    assert pos_sentiment['sentiment'] in ['Positive', 'Negative', 'Neutral']
    assert neg_sentiment['sentiment'] in ['Positive', 'Negative', 'Neutral']
    assert neutral_sentiment['sentiment'] in ['Positive', 'Negative', 'Neutral']
    
    # Test polarity values
    assert -1 <= pos_sentiment['polarity'] <= 1
    assert -1 <= neg_sentiment['polarity'] <= 1
    assert -1 <= neutral_sentiment['polarity'] <= 1

def test_extract_keywords():
    """Test the keyword extraction function"""
    from main import extract_keywords
    text = "This is a sample text with some important keywords like technology, development, and innovation."
    keywords = extract_keywords(text, num_keywords=3)
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert all(isinstance(keyword, str) for keyword in keywords)

def test_get_article_insights():
    """Test the article insights function"""
    from main import get_article_insights
    text = "This is a test article with multiple sentences. Each sentence should contribute to the word count. This is the third sentence."
    insights = get_article_insights(text)
    
    assert 'word_count' in insights
    assert 'sentence_count' in insights
    assert 'keywords' in insights
    assert 'reading_time_minutes' in insights
    
    assert isinstance(insights['word_count'], int)
    assert isinstance(insights['sentence_count'], int)
    assert isinstance(insights['keywords'], list)
    assert isinstance(insights['reading_time_minutes'], int)

def test_empty_text_handling():
    """Test handling of empty text"""
    from main import generate_summary, analyze_sentiment, extract_keywords, get_article_insights
    empty_summary = generate_summary("")
    empty_sentiment = analyze_sentiment("")
    empty_keywords = extract_keywords("")
    empty_insights = get_article_insights("")
    
    assert isinstance(empty_summary, str)
    assert isinstance(empty_sentiment, dict)
    assert isinstance(empty_keywords, list)
    assert isinstance(empty_insights, dict)