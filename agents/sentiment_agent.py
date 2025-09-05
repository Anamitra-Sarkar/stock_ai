import random

class SentimentAgent:
    """
    Sentiment Agent: Analyzes news sentiment using basic keyword analysis
    """

    def __init__(self):
        self.positive_words = [
            'growth', 'profit', 'increase', 'strong', 'upgrade', 'bullish',
            'positive', 'gain', 'rise', 'beat', 'outperform', 'surge',
            'rally', 'boom', 'breakthrough', 'success', 'expansion'
        ]

        self.negative_words = [
            'loss', 'decline', 'decrease', 'weak', 'downgrade', 'bearish',
            'negative', 'drop', 'fall', 'miss', 'underperform', 'crash',
            'plunge', 'recession', 'concern', 'warning', 'layoff'
        ]

    def analyze_sentiment(self, news_text: str) -> str:
        """
        Analyzes sentiment of news text using keyword matching
        """
        if not news_text:
            return "Neutral"

        news_lower = news_text.lower()

        positive_count = sum(1 for word in self.positive_words if word in news_lower)
        negative_count = sum(1 for word in self.negative_words if word in news_lower)

        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"