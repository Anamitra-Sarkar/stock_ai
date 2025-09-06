class AlertAgent:
    """
    Alert Agent: Generates alerts based on stock analysis
    """

    def check_for_alerts(self, stock_analysis: dict) -> list:
        """
        Checks for various alert conditions based on stock analysis
        """
        alerts = []
        ticker = stock_analysis['ticker']
        prediction = stock_analysis['prediction']
        sentiment = stock_analysis['sentiment']

        # High confidence prediction alert
        if prediction['confidence'] > 90:
            alerts.append({
                'type': 'high_confidence',
                'ticker': ticker,
                'message': (f"High confidence ({prediction['confidence']}%) prediction for {ticker}: "
                          f"{prediction['trend']} trend expected")
            })

        # Sentiment mismatch alert
        if sentiment == 'Negative' and prediction['trend'] == 'up':
            alerts.append({
                'type': 'sentiment_mismatch',
                'ticker': ticker,
                'message': f"Warning: Negative sentiment but upward trend predicted for {ticker}"
            })

        # Strong buy signal
        if sentiment == 'Positive' and prediction['trend'] == 'up' and prediction['confidence'] > 85:
            alerts.append({
                'type': 'buy_signal',
                'ticker': ticker,
                'message': f"Strong buy signal for {ticker}: Positive sentiment + high confidence upward trend"
            })

        return alerts
