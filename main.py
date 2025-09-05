import os
import asyncio
import aiohttp
from flask import Flask, jsonify, request
from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from agents.sentiment_agent import SentimentAgent
from agents.advisor_agent import AdvisorAgent
from agents.alert_agent import AlertAgent

# --- INITIALIZE THE APPLICATION AND AGENTS ---
app = Flask(__name__)

# Production-ready: Load API key from environment variables.
# On Replit, use the "Secrets" tab to set this key.
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("API Key not found. Please set ALPHA_VANTAGE_API_KEY in your environment secrets.")

# Create singleton instances of our agents
data_agent = DataAgent(api_key=ALPHA_VANTAGE_API_KEY)
prediction_agent = PredictionAgent()
sentiment_agent = SentimentAgent()
advisor_agent = AdvisorAgent()
alert_agent = AlertAgent()

# --- ASYNC HELPER FUNCTION ---
async def analyze_stock_async(session, ticker):
    """A helper function to run a full analysis on a single stock asynchronously."""
    stock_data_task = data_agent.get_current_stock_data_async(session, ticker)
    news_task = data_agent.get_latest_news_async(session, ticker)

    stock_data, news = await asyncio.gather(stock_data_task, news_task)

    if not stock_data or stock_data['price'] == 0:
        return None

    prediction = prediction_agent.predict_trend(ticker, stock_data['price'])
    sentiment = sentiment_agent.analyze_sentiment(news)

    return {
        "ticker": ticker,
        "name": stock_data.get('name', ticker),
        "price": round(stock_data['price'], 2),
        "news": news,
        "prediction": prediction,
        "sentiment": sentiment
    }

# --- API ENDPOINTS ---

@app.route('/api/dashboard', methods=['GET'])
async def get_dashboard_data():
    """
    High-performance endpoint using asyncio to fetch all data concurrently.
    """
    tickers = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'META']
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_stock_async(session, ticker) for ticker in tickers]
        all_stock_analysis = await asyncio.gather(*tasks)

    # Filter out None results and check for alerts
    valid_analyses = [analysis for analysis in all_stock_analysis if analysis]
    all_alerts = []
    for analysis in valid_analyses:
        all_alerts.extend(alert_agent.check_for_alerts(analysis))

    return jsonify({
        "stocks": valid_analyses,
        "alerts": all_alerts
    })

@app.route('/api/ask', methods=['POST'])
async def ask_assistant():
    """
    Upgraded endpoint that can handle any ticker symbol provided in the query.
    """
    data = request.json
    user_query = data.get('query', '').lower()
    user_profile = data.get('profile', {})

    # Use regex or a more robust method in the future to extract tickers
    # For now, we split the query and look for potential tickers (e.g., all caps)
    potential_tickers = [word.upper() for word in user_query.split() if len(word) > 1 and len(word) < 6 and word.isalpha()]

    if potential_tickers:
        # Analyze the first potential ticker found
        ticker_to_analyze = potential_tickers[0]
        async with aiohttp.ClientSession() as session:
            analysis = await analyze_stock_async(session, ticker_to_analyze)

        if analysis:
            response = advisor_agent.generate_recommendation(analysis, user_profile)
            return jsonify({"response": response})
        else:
            return jsonify({"response": f"Sorry, I could not find data for {ticker_to_analyze}."}), 404

    if 'sip' in user_query or 'systematic investment plan' in user_query:
        response = advisor_agent.plan_sip(user_profile)
        return jsonify({"response": response})

    return jsonify({"response": "I can provide analysis on any stock (e.g., 'what do you think about NVDA?') or plan a SIP. How can I help?"})