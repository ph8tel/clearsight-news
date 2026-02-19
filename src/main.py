from news_insight_app import create_app
from news_insight_app.services import (
    analyze_sentiment,
    extract_keywords,
    generate_summary,
    get_article_insights,
)

app = create_app()

if __name__ == '__main__':
    # Run the app on all network interfaces to serve on local network
    app.run(host='0.0.0.0', port=5000, debug=True)
