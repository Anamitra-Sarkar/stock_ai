import os
import requests
import json
import time
from flask import Flask, jsonify, request
from agents.prediction_agent import PredictionAgent
from agents.sentiment_agent import SentimentAgent
from agents.advisor_agent import AdvisorAgent
from agents.alert_agent import AlertAgent

# --- INITIALIZE THE APPLICATION AND AGENTS ---
app = Flask(__name__)

# Get API key
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

if not ALPHA_VANTAGE_API_KEY:
    print("🚨 WARNING: No API key found!")

# Create instances
prediction_agent = PredictionAgent()
sentiment_agent = SentimentAgent()
advisor_agent = AdvisorAgent()
alert_agent = AlertAgent()

# --- DIVERSE STOCK UNIVERSE ---
STOCK_CATEGORIES = {
    'Technology': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG'],
    'Consumer': ['KO', 'PG', 'WMT', 'HD'],
    'Industrial': ['GE', 'CAT', 'BA', 'MMM'],
    'Telecom': ['VZ', 'T', 'TMUS', 'CMCSA']
}

# Flatten all tickers
ALL_TICKERS = []
for category, tickers in STOCK_CATEGORIES.items():
    ALL_TICKERS.extend(tickers)

# Rate limiting
api_call_count = 0
last_reset_time = time.time()
MAX_CALLS_PER_HOUR = 20  # Conservative limit

def can_make_api_call():
    global api_call_count, last_reset_time

    # Reset counter every hour
    current_time = time.time()
    if current_time - last_reset_time > 3600:  # 1 hour
        api_call_count = 0
        last_reset_time = current_time

    return api_call_count < MAX_CALLS_PER_HOUR

def increment_api_count():
    global api_call_count
    api_call_count += 1

# --- SERVE FRONTEND ---
@app.route('/')
def serve_frontend():
    try:
        with open('index.html', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Frontend file not found", 404

# --- REAL DATA FUNCTIONS ---
def get_real_stock_data(ticker):
    """Get REAL stock data with rate limiting"""
    if not ALPHA_VANTAGE_API_KEY:
        print(f"⚠️  No API key - returning mock data for {ticker}")
        return get_mock_stock_data(ticker)

    if not can_make_api_call():
        print(f"⏰ Rate limit reached - using mock data for {ticker}")
        return get_mock_stock_data(ticker)

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }

        print(f"🔄 Fetching REAL data for {ticker}... (Call #{api_call_count + 1})")
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        increment_api_count()

        # Check for rate limit message
        if 'Information' in data and 'rate limit' in data['Information'].lower():
            print(f"⚠️  Rate limit hit for {ticker}")
            return get_mock_stock_data(ticker)

        if 'Global Quote' in data and data['Global Quote']:
            quote = data['Global Quote']
            price = float(quote.get('05. price', 0))

            if price > 0:
                print(f"✅ REAL data for {ticker}: ${price}")
                return {
                    'name': ticker,
                    'price': round(price, 2),
                    'change': round(float(quote.get('09. change', 0)), 2),
                    'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                    'is_real': True,
                    'category': get_stock_category(ticker)
                }

        print(f"❌ No valid data for {ticker}")
        return get_mock_stock_data(ticker)

    except Exception as e:
        print(f"🚨 ERROR fetching real data for {ticker}: {e}")
        return get_mock_stock_data(ticker)

def get_stock_category(ticker):
    """Get the category/sector of a stock"""
    for category, tickers in STOCK_CATEGORIES.items():
        if ticker in tickers:
            return category
    return "Other"

def get_mock_stock_data(ticker):
    """Realistic mock data based on actual stock patterns"""
    import random
    random.seed(hash(ticker))

    # More realistic price ranges based on categories
    category = get_stock_category(ticker)

    if category == 'Technology':
        price = round(100 + random.uniform(50, 400), 2)
    elif category == 'Healthcare':
        price = round(80 + random.uniform(20, 200), 2)
    elif category == 'Finance':
        price = round(30 + random.uniform(10, 150), 2)
    elif category == 'Energy':
        price = round(40 + random.uniform(20, 180), 2)
    else:
        price = round(50 + random.uniform(20, 250), 2)

    return {
        'name': f"{ticker} ({category})",
        'price': price,
        'change': round(random.uniform(-5, 5), 2),
        'change_percent': f"{random.uniform(-3, 3):.2f}",
        'is_real': False,
        'category': category
    }

def analyze_stock(ticker):
    """Analyze a single stock"""
    stock_data = get_real_stock_data(ticker)

    if not stock_data or stock_data['price'] == 0:
        return None

    prediction = prediction_agent.predict_trend(ticker, stock_data['price'])

    # Simple sentiment based on category and price movement
    if stock_data.get('change', 0) > 2:
        sentiment = "Positive"
    elif stock_data.get('change', 0) < -2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    news = f"{ticker} ({stock_data['category']}) {'gains' if stock_data.get('change', 0) > 0 else 'declines'} in today's trading"

    return {
        "ticker": ticker,
        "name": stock_data.get('name', ticker),
        "price": stock_data['price'],
        "category": stock_data['category'],
        "change": stock_data.get('change', 0),
        "change_percent": stock_data.get('change_percent', '0'),
        "news": news,
        "prediction": prediction,
        "sentiment": sentiment,
        "is_real_data": stock_data.get('is_real', False)
    }

# --- API ENDPOINTS ---

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get diversified dashboard with multiple sectors"""
    print("🎯 Dashboard API called - fetching diversified portfolio...")

    # Select a diverse mix of stocks (2 from each major category)
    selected_stocks = []
    for category, tickers in STOCK_CATEGORIES.items():
        selected_stocks.extend(tickers[:2])  # Take first 2 from each category

    # Limit to 10 stocks to avoid rate limits
    selected_stocks = selected_stocks[:10]

    all_stock_analysis = []
    real_data_count = 0

    for ticker in selected_stocks:
        analysis = analyze_stock(ticker)
        if analysis:
            all_stock_analysis.append(analysis)
            if analysis.get('is_real_data'):
                real_data_count += 1

    # Generate alerts
    all_alerts = []
    for analysis in all_stock_analysis:
        all_alerts.extend(alert_agent.check_for_alerts(analysis))

    print(f"📊 Dashboard: {real_data_count}/{len(all_stock_analysis)} stocks have REAL data")
    print(f"⏰ API calls used: {api_call_count}/{MAX_CALLS_PER_HOUR}")

    return jsonify({
        "stocks": all_stock_analysis,
        "alerts": all_alerts,
        "data_status": {
            "total_stocks": len(all_stock_analysis),
            "real_data_count": real_data_count,
            "api_calls_used": api_call_count,
            "api_calls_remaining": MAX_CALLS_PER_HOUR - api_call_count,
            "categories_covered": list(STOCK_CATEGORIES.keys())
        }
    })

@app.route('/api/category/<category>', methods=['GET'])
def get_category_stocks(category):
    """Get stocks from a specific category"""
    if category not in STOCK_CATEGORIES:
        return jsonify({"error": "Category not found"}), 404

    tickers = STOCK_CATEGORIES[category]
    stocks = []

    for ticker in tickers:
        analysis = analyze_stock(ticker)
        if analysis:
            stocks.append(analysis)

    return jsonify({
        "category": category,
        "stocks": stocks
    })

@app.route('/api/search/<ticker>', methods=['GET'])
def search_stock(ticker):
    """Search for any specific stock"""
    ticker = ticker.upper()
    print(f"🔍 Searching for {ticker}...")

    analysis = analyze_stock(ticker)
    if analysis:
        return jsonify({"stock": analysis})
    else:
        return jsonify({"error": f"Could not find data for {ticker}"}), 404

@app.route('/api/ask', methods=['POST'])
def ask_assistant():
    """Enhanced chat with category awareness"""
    data = request.json
    user_query = data.get('query', '').lower()
    user_profile = data.get('profile', {})

    # Check for category requests
    for category in STOCK_CATEGORIES.keys():
        if category.lower() in user_query:
            tickers = STOCK_CATEGORIES[category][:3]  # Limit to 3 to save API calls
            response = f"Here are some top {category} stocks:\n\n"

            for ticker in tickers:
                analysis = analyze_stock(ticker)
                if analysis:
                    response += f"• {ticker}: ${analysis['price']} ({analysis['sentiment']} sentiment)\n"

            return jsonify({"response": response})

    # Extract tickers
    potential_tickers = [word.upper() for word in user_query.split() 
                        if len(word) > 1 and len(word) < 6 and word.isalpha()]

    if potential_tickers:
        ticker = potential_tickers[0]
        analysis = analyze_stock(ticker)
        if analysis:
            response = advisor_agent.generate_recommendation(analysis, user_profile)
            response += f"\n\n💡 {ticker} is in the {analysis['category']} sector."

            if analysis.get('is_real_data'):
                response += f"\n✅ Real-time data: ${analysis['price']}"
            else:
                response += f"\n⚠️  Using estimated data (API limit reached)"

            return jsonify({"response": response})

    return jsonify({"response": "I can analyze stocks across Technology, Healthcare, Finance, Energy, Consumer, Industrial, and Telecom sectors. Try asking about a specific stock or sector!"})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Enhanced status with rate limiting info"""
    return jsonify({
        "api_key_configured": bool(ALPHA_VANTAGE_API_KEY),
        "api_calls_used": api_call_count,
        "api_calls_remaining": MAX_CALLS_PER_HOUR - api_call_count,
        "categories_available": list(STOCK_CATEGORIES.keys()),
        "total_stocks_available": len(ALL_TICKERS)
    })

if __name__ == '__main__':
    print("🚀 Starting REAL Stock AI with diverse portfolio...")
    print(f"📊 Covering {len(STOCK_CATEGORIES)} sectors with {len(ALL_TICKERS)} stocks")
    print(f"🔑 API Key: {'Configured' if ALPHA_VANTAGE_API_KEY else 'Missing'}")
    print(f"⏰ Rate limit: {MAX_CALLS_PER_HOUR} calls per hour")
    app.run(host='0.0.0.0', port=5000, debug=True)