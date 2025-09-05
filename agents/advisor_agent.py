class AdvisorAgent:
    """
    Advisor Agent: Synthesizes data to allocate money, suggest SIPs, and explain reasoning.
    This agent uses a rule-based system. A more advanced version could use a
    reinforcement learning model to optimize portfolio allocation based on user goals.
    """
    def generate_recommendation(self, stock_analysis, user_profile):
        """
        Generates a user-friendly investment recommendation based on all available data.
        """
        ticker = stock_analysis['ticker']
        confidence = stock_analysis['prediction']['confidence']
        sentiment = stock_analysis['sentiment']
        trend = stock_analysis['prediction']['trend']
        news = stock_analysis['news']

        # Rule-based logic for advice
        advice_parts = [
            f"Regarding {ticker}, my analysis shows a confidence score of {confidence}% for an '{trend}' trend.",
            f"The news sentiment is currently '{sentiment}', driven by the headline: \"{news}\"."
        ]

        if confidence > 90 and sentiment == 'Positive' and trend == 'up':
            advice_parts.append(
                "This indicates a strong potential opportunity. Given your financial profile, a small, "
                "calculated investment could be considered. For example, allocating 2-5% of your investable capital."
            )
        elif sentiment == 'Negative' or trend == 'down':
            advice_parts.append(
                "Caution is advised. The current indicators are not favorable for investment. "
                "It might be better to observe for now or consider defensive positions if you already hold this stock."
            )
        else:
            advice_parts.append(
                "The signals are mixed, suggesting a moderate risk scenario. "
                "It would be wise to wait for a clearer trend to emerge before committing capital."
            )

        advice_parts.append("Remember, this is a model-driven analysis, not financial advice.")

        return " ".join(advice_parts)

    def plan_sip(self, user_profile):
        """
        Generates a simple SIP recommendation.
        """
        investable_capital = user_profile.get('initialCapital', 10000)
        monthly_investment = (investable_capital * 0.2) / 12  # Example: 20% of capital per year

        return (
            f"A Systematic Investment Plan (SIP) is a great strategy for long-term growth. "
            f"Based on your profile, I'd suggest starting with a stock with strong fundamentals, like MSFT. "
            f"You could consider allocating around ${monthly_investment:.2f} per month. "
            f"This approach helps manage risk and leverages dollar-cost averaging."
        )
