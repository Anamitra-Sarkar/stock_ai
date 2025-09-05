import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAgent:
    """
    Sentiment Agent: Analyzes news and social media to assess market sentiment.
    This agent uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner),
    which is a pre-trained model tuned for social media text.
    It's a good starting point before building a custom transformer-based model.
    """
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            print("Downloading NLTK VADER lexicon...")
            nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """
        Analyzes a piece of text (e.g., a news headline) and returns its sentiment.
        """
        # Get polarity scores
        scores = self.analyzer.polarity_scores(text)

        # Classify based on the compound score
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
