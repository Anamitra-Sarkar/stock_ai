import os
import requests
import json
import time
import asyncio
import numpy as np
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

# Import enterprise modules
from config import config
from agents.prediction_agent import PredictionAgent
from agents.sentiment_agent import SentimentAgent
from agents.advisor_agent import AdvisorAgent
from agents.alert_agent import AlertAgent
from streaming.websocket_manager import StreamingManager
from cache.redis_cache import cache_manager
from database.connection import db_manager
from portfolio.optimizer import PortfolioOptimizer
from risk.risk_manager import RiskManager

# --- INITIALIZE ENTERPRISE APPLICATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = config.secret_key

# Enable CORS for all routes
CORS(app, cors_allowed_origins="*")

# Initialize WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize enterprise services
streaming_manager = StreamingManager(socketio)
portfolio_optimizer = PortfolioOptimizer()
risk_manager = RiskManager()

print("🚀 Initializing Enterprise Stock AI Platform...")

# Initialize core services asynchronously
async def init_enterprise_services():
    """Initialize all enterprise services"""
    try:
        # Initialize cache
        await cache_manager.initialize()
        
        # Initialize database (optional, will work without)
        try:
            await db_manager.initialize()
        except Exception as e:
            print(f"⚠️  Database initialization skipped: {e}")
        
        print("✅ Enterprise services initialized")
    except Exception as e:
        print(f"❌ Service initialization error: {e}")

# Check deployment environment
MINIMAL_STARTUP = os.getenv('MINIMAL_STARTUP', 'false').lower() == 'true'
SKIP_ML_TRAINING = os.getenv('SKIP_ML_TRAINING', 'false').lower() == 'true'

# Run initialization
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(init_enterprise_services())

# Get API key from config
ALPHA_VANTAGE_API_KEY = config.api.alpha_vantage_key

if not ALPHA_VANTAGE_API_KEY:
    print("🚨 WARNING: No Alpha Vantage API key configured!")

# Create AI agents with enterprise features
prediction_agent = PredictionAgent()
sentiment_agent = SentimentAgent()
advisor_agent = AdvisorAgent()
alert_agent = AlertAgent()

# --- ENHANCED STOCK UNIVERSE ---
STOCK_CATEGORIES = {
    'Technology': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META', 'TSLA', 'CRM', 'ORCL'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BK', 'USB'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX'],
    'Consumer': ['KO', 'PG', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS'],
    'Industrial': ['GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'LMT', 'RTX'],
    'Telecom': ['VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'DIS', 'NFLX', 'GOOGL']
}

# Flatten all tickers
ALL_TICKERS = []
for category, tickers in STOCK_CATEGORIES.items():
    ALL_TICKERS.extend(tickers)

# Enhanced rate limiting with Redis support
api_call_count = 0
last_reset_time = time.time()
MAX_CALLS_PER_HOUR = config.api.rate_limit_per_hour

async def can_make_api_call():
    """Enhanced rate limiting with Redis tracking"""
    global api_call_count, last_reset_time
    
    # Check Redis cache for rate limit status
    cache_key = "api_rate_limit"
    cached_count = await cache_manager.get(cache_key)
    
    current_time = time.time()
    
    # Reset counter every hour
    if current_time - last_reset_time > 3600:
        api_call_count = 0
        last_reset_time = current_time
        await cache_manager.set(cache_key, 0, 3600)
        return True
    
    # Use cached count if available
    if cached_count is not None:
        api_call_count = cached_count
    
    return api_call_count < MAX_CALLS_PER_HOUR

async def increment_api_count():
    """Increment API call counter with Redis sync"""
    global api_call_count
    api_call_count += 1
    
    cache_key = "api_rate_limit"
    await cache_manager.set(cache_key, api_call_count, 3600)

# Legacy sync functions for backward compatibility
def can_make_api_call_sync():
    """Legacy sync version of rate limiting"""
    global api_call_count, last_reset_time
    
    current_time = time.time()
    if current_time - last_reset_time > 3600:
        api_call_count = 0
        last_reset_time = current_time
    
    return api_call_count < MAX_CALLS_PER_HOUR

def increment_api_count_sync():
    """Legacy sync version of increment"""
    global api_call_count
    api_call_count += 1
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

    if not can_make_api_call_sync():
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

        increment_api_count_sync()

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
    """Enhanced status with enterprise features"""
    cache_stats = cache_manager.get_stats()
    model_status = prediction_agent.get_model_status()
    
    return jsonify({
        "api_key_configured": bool(ALPHA_VANTAGE_API_KEY),
        "api_calls_used": api_call_count,
        "api_calls_remaining": MAX_CALLS_PER_HOUR - api_call_count,
        "categories_available": list(STOCK_CATEGORIES.keys()),
        "total_stocks_available": len(ALL_TICKERS),
        "cache_stats": cache_stats,
        "ml_models": model_status,
        "streaming_active": streaming_manager.is_streaming,
        "services": {
            "cache": cache_stats['backend'],
            "database": "Available" if db_manager._initialized else "Unavailable",
            "websocket": "Active",
            "ml_models": model_status['total_models']
        }
    })

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])  
def health_check():
    """Basic health check for load balancers and monitoring"""
    try:
        # Basic health indicators
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - start_time if 'start_time' in globals() else 0,
            "version": "1.0.0",
            "environment": os.getenv('FLASK_ENV', 'development')
        }
        
        # Check critical services
        services = {}
        
        # Cache health
        try:
            cache_health = asyncio.run(cache_manager.health_check())
            services['cache'] = cache_health['status']
        except:
            services['cache'] = 'unavailable'
        
        # Database health  
        services['database'] = 'available' if db_manager._initialized else 'unavailable'
        
        # Check if any critical service is down
        critical_down = services['cache'] == 'unavailable'
        
        if critical_down:
            health_status['status'] = 'degraded'
            
        health_status['services'] = services
        
        return jsonify(health_status), 200 if health_status['status'] == 'healthy' else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": time.time()
        }), 503

# --- ENTERPRISE API ENDPOINTS ---

@app.route('/api/prediction/<ticker>', methods=['GET'])
def get_prediction(ticker):
    """Get ML prediction for a stock"""
    ticker = ticker.upper()
    current_price = request.args.get('price', type=float)
    
    if not current_price:
        # Try to get current price from cache or mock
        current_price = 100.0  # Default fallback
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prediction = loop.run_until_complete(
            prediction_agent.predict_trend(ticker, current_price)
        )
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/technical/<ticker>', methods=['GET'])
def get_technical_analysis(ticker):
    """Get technical indicators for a stock"""
    ticker = ticker.upper()
    timeframe = request.args.get('timeframe', '1d')
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get cached technical indicators
        indicators = loop.run_until_complete(
            cache_manager.get_technical_indicators(ticker, timeframe)
        )
        
        if indicators:
            return jsonify({
                "ticker": ticker,
                "timeframe": timeframe,
                "indicators": indicators,
                "cached": True
            })
        else:
            return jsonify({
                "ticker": ticker,
                "timeframe": timeframe,
                "indicators": {},
                "cached": False,
                "message": "No cached data available"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio allocation"""
    data = request.json
    holdings = data.get('holdings', [])
    objective = data.get('objective', 'sharpe')  # sharpe, min_volatility, max_return
    
    if not holdings:
        return jsonify({"error": "No holdings provided"}), 400
    
    try:
        # Prepare price data (in production, get from database)
        price_data = {}
        for holding in holdings:
            symbol = holding['symbol']
            # Mock price data for optimization
            price_data[symbol] = [
                {"timestamp": f"2023-{i:02d}-01", "close_price": 100 + i + (hash(symbol) % 50)}
                for i in range(1, 101)
            ]
        
        optimization_result = portfolio_optimizer.optimize_portfolio(price_data, objective)
        
        return jsonify({
            "optimization": optimization_result,
            "objective": objective,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/risk', methods=['POST'])
def analyze_portfolio_risk():
    """Analyze portfolio risk metrics"""
    data = request.json
    holdings = data.get('holdings', [])
    
    if not holdings:
        return jsonify({"error": "No holdings provided"}), 400
    
    try:
        # Generate mock price history for risk analysis
        price_history = {}
        for holding in holdings:
            symbol = holding['symbol']
            # Mock historical prices
            base_price = 100
            prices = [base_price]
            for _ in range(252):  # 1 year of daily data
                change = np.random.normal(0, 0.02)
                new_price = prices[-1] * (1 + change)
                prices.append(max(1.0, new_price))
            price_history[symbol] = prices[1:]
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(holdings, price_history)
        position_risks = risk_manager.analyze_position_risk(holdings, sum(h.get('market_value', 0) for h in holdings))
        compliance_check = risk_manager.check_risk_limits(holdings, risk_metrics)
        
        return jsonify({
            "risk_metrics": {
                "var_95": risk_metrics.value_at_risk_95,
                "var_99": risk_metrics.value_at_risk_99,
                "max_drawdown": risk_metrics.max_drawdown,
                "volatility": risk_metrics.volatility,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio
            },
            "position_risks": [
                {
                    "symbol": pr.symbol,
                    "portfolio_weight": pr.portfolio_weight,
                    "concentration_score": pr.concentration_score,
                    "liquidity_score": pr.liquidity_score
                } for pr in position_risks
            ],
            "compliance": compliance_check,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/multi-timeframe/<ticker>', methods=['GET'])
def get_multi_timeframe_analysis(ticker):
    """Get multi-timeframe analysis for a stock"""
    ticker = ticker.upper()
    current_price = request.args.get('price', 100.0, type=float)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        multi_tf_analysis = loop.run_until_complete(
            prediction_agent.get_multi_timeframe_analysis(ticker, current_price)
        )
        
        return jsonify({
            "ticker": ticker,
            "multi_timeframe_analysis": multi_tf_analysis,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/streaming/stats', methods=['GET'])
def get_streaming_stats():
    """Get real-time streaming statistics"""
    return jsonify(streaming_manager.get_subscription_stats())

@app.route('/api/market/summary', methods=['GET'])
def get_market_summary():
    """Get current market summary"""
    return jsonify(streaming_manager.get_market_summary())

# --- WEBSOCKET EVENT HANDLERS ---

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    print(f"📡 Client {client_id} connected")
    emit('connection_established', {
        'client_id': client_id,
        'message': 'Connected to Stock AI streaming service',
        'available_rooms': ['dashboard', 'portfolio', 'alerts']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    print(f"📡 Client {client_id} disconnected")
    
    # Clean up subscriptions
    streaming_manager.unsubscribe_client(client_id)

@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription to real-time data"""
    client_id = request.sid
    symbols = data.get('symbols', [])
    room = data.get('room', 'default')
    
    # Join the room
    join_room(room)
    
    # Subscribe to streaming
    success = streaming_manager.subscribe_to_symbol(client_id, symbols, room)
    
    if success:
        emit('subscription_confirmed', {
            'symbols': symbols,
            'room': room,
            'message': f'Subscribed to {len(symbols)} symbols in room {room}'
        })
        
        # Send initial data for subscribed symbols
        initial_data = {}
        for symbol in symbols:
            details = streaming_manager.get_symbol_details(symbol)
            if details:
                initial_data[symbol] = details
        
        if initial_data:
            emit('market_data', initial_data, room=room)
    else:
        emit('subscription_error', {
            'error': 'Failed to subscribe to symbols',
            'symbols': symbols
        })

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Handle unsubscription from real-time data"""
    client_id = request.sid
    room = data.get('room', 'default')
    
    # Leave the room
    leave_room(room)
    
    # Unsubscribe from streaming
    streaming_manager.unsubscribe_client(client_id, room)
    
    emit('unsubscription_confirmed', {
        'room': room,
        'message': f'Unsubscribed from room {room}'
    })

@socketio.on('get_prediction')
def handle_get_prediction(data):
    """Handle real-time prediction request"""
    ticker = data.get('ticker', '').upper()
    current_price = data.get('price', 100.0)
    
    if not ticker:
        emit('prediction_error', {'error': 'Ticker symbol required'})
        return
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        prediction = loop.run_until_complete(
            prediction_agent.predict_trend(ticker, current_price)
        )
        
        emit('prediction_result', {
            'ticker': ticker,
            'prediction': prediction,
            'timestamp': time.time()
        })
    except Exception as e:
        emit('prediction_error', {
            'ticker': ticker,
            'error': str(e)
        })

# --- APPLICATION STARTUP ---

def start_enterprise_services():
    """Start all enterprise background services"""
    print("🚀 Starting enterprise services...")
    
    # Start real-time streaming
    streaming_manager.start_streaming()
    
    print("✅ Enterprise services started")

if __name__ == '__main__':
    # Initialize start time for health checks
    start_time = time.time()
    
    # Start enterprise services
    start_enterprise_services()
    
    print("🚀 Starting Enterprise Stock AI Platform...")
    print(f"📊 Covering {len(STOCK_CATEGORIES)} sectors with {len(ALL_TICKERS)} stocks")
    print(f"🔑 API Key: {'Configured' if ALPHA_VANTAGE_API_KEY else 'Missing'}")
    print(f"⏰ Rate limit: {MAX_CALLS_PER_HOUR} calls per hour")
    
    # Show deployment mode status
    if MINIMAL_STARTUP:
        print("🚀 Running in MINIMAL_STARTUP mode (deployment-optimized)")
    if SKIP_ML_TRAINING:
        print("🧠 ML training SKIPPED (deployment mode)")
        
    print(f"🏗️  Enterprise features: WebSocket, ML, Portfolio Optimization, Risk Management")
    print(f"📡 WebSocket streaming: Active")
    print(f"🗄️  Database: {'Connected' if db_manager._initialized else 'Using fallback'}")
    print(f"🚀 Cache: {cache_manager.get_stats()['backend']}")
    
    # Make start_time available globally for health checks
    globals()['start_time'] = start_time
    
    print(f"🌐 Health check available at: http://{config.host}:{config.port}/health")
    
    # Run with SocketIO support
    socketio.run(
        app, 
        host=config.host, 
        port=config.port, 
        debug=config.debug,
        allow_unsafe_werkzeug=True
    )