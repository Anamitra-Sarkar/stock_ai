import random

class AlertAgent:
    """
    Alert Agent: Monitors market conditions and notifies users of risks/opportunities.
    This agent checks for predefined conditions in the data stream.
    """
    def check_for_alerts(self, stock_analysis):
        """
        Checks a single stock's analysis for alert-worthy conditions.
        """
        alerts = []
        ticker = stock_analysis['ticker']
        confidence = stock_analysis['prediction']['confidence']
        sentiment = stock_analysis['sentiment']

        # Rule 1: High-confidence opportunity
        if confidence > 92 and sentiment == "Positive":
            alerts.append({
                "type": "opportunity",
                "message": f"{ticker} shows strong buy signals. Prediction confidence is at {confidence}% with positive news."
            })

        # Rule 2: High-risk situation
        if sentiment == "Negative":
            alerts.append({
                "type": "risk",
                "message": f"Sentiment for {ticker} has turned negative due to recent news. Monitor your positions closely."
            })

        # Rule 3: Significant price movement (simulated)
        if random.random() < 0.1: # Simulate a 10% chance of a major price event
             alerts.append({
                "type": "price",
                "message": f"{ticker} just broke a key technical resistance/support level. Increased volatility expected."
            })

        return alerts
