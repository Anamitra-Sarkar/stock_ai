# Enterprise Stock AI Platform

An enterprise-grade fintech platform for stock market analysis, prediction, and portfolio management using advanced machine learning and real-time streaming capabilities.

## 🚀 Features

### Core Features
- **LSTM Neural Networks** for advanced price prediction with fallback support
- **Real-time WebSocket streaming** for live market data
- **Portfolio optimization** using Modern Portfolio Theory
- **20+ Technical indicators** (RSI, MACD, Bollinger Bands, SMA, EMA, ATR, etc.)
- **Risk management** with comprehensive metrics and stress testing
- **Multi-timeframe analysis** (1m, 5m, 1h, 1d, 1w)
- **Enterprise caching** with Redis and in-memory fallback
- **PostgreSQL database** for data persistence

### Enterprise Features
- **Docker containerization** for easy deployment
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive testing** suite
- **Security scanning** with Bandit and Safety
- **Performance monitoring** and metrics
- **Environment-based configuration**
- **Database migrations** and initialization
- **API documentation** and OpenAPI specs

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)
- Node.js 18+ (for frontend development)

## 🛠️ Installation

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Anamitra-Sarkar/stock_ai.git
cd stock_ai

# Copy environment variables
cp .env.example .env

# Edit .env with your API keys and configuration
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f stock-ai-app
```

### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/Anamitra-Sarkar/stock_ai.git
cd stock_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ALPHA_VANTAGE_API_KEY=your_api_key_here
export DB_HOST=localhost
export REDIS_HOST=localhost

# Initialize database (if using PostgreSQL)
python -c "
import asyncio
from database import db_manager
asyncio.run(db_manager.initialize())
"

# Start the application
python main.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_ai
DB_USER=postgres
DB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Application Configuration
DEBUG=false
SECRET_KEY=your-super-secret-key
HOST=0.0.0.0
PORT=5000

# ML Configuration
LSTM_SEQUENCE_LENGTH=60
LSTM_EPOCHS=50
LSTM_BATCH_SIZE=32
MODEL_SAVE_PATH=./models

# Rate Limiting
API_RATE_LIMIT=500
```

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   WebSocket     │    │   REST API      │
│   (HTML/JS)     │◄──►│   Streaming     │◄──►│   Endpoints     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────┐
                    │   Flask App     │
                    │   (main.py)     │
                    └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Agents    │    │   ML Models │    │  Services   │
│  - Prediction │    │  - LSTM     │    │  - Cache    │
│  - Advisory   │    │  - Linear   │    │  - Database │
│  - Alert      │    │  - Technical│    │  - Risk     │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌─────────────────┐
                │   Data Layer    │
                │  - PostgreSQL   │
                │  - Redis Cache  │
                └─────────────────┘
```

### Module Structure
```
stock_ai/
├── agents/                 # AI Agents
│   ├── prediction_agent.py # LSTM-based predictions
│   ├── advisor_agent.py    # Investment advice
│   ├── alert_agent.py      # Trading alerts
│   └── sentiment_agent.py  # News sentiment
├── ml_models/              # Machine Learning
│   └── lstm_predictor.py   # LSTM neural networks
├── indicators/             # Technical Analysis
│   └── technical_indicators.py # 20+ indicators
├── portfolio/              # Portfolio Management
│   └── optimizer.py        # Modern Portfolio Theory
├── risk/                   # Risk Management
│   └── risk_manager.py     # Comprehensive risk metrics
├── streaming/              # Real-time Data
│   └── websocket_manager.py # WebSocket streaming
├── database/               # Data Layer
│   ├── models.py          # Database models
│   └── connection.py      # Connection management
├── cache/                  # Caching Layer
│   └── redis_cache.py     # Redis with fallback
├── config.py              # Configuration management
├── main.py                # Flask application
└── requirements.txt       # Python dependencies
```

## 📡 API Documentation

### REST API Endpoints

#### Market Data
- `GET /api/dashboard` - Get diversified market dashboard
- `GET /api/category/{category}` - Get stocks by category
- `GET /api/search/{ticker}` - Search specific stock
- `GET /api/status` - System status and health

#### AI Analysis
- `POST /api/ask` - Chat with AI advisor
- `GET /api/prediction/{ticker}` - Get ML predictions
- `GET /api/technical/{ticker}` - Technical indicators
- `GET /api/risk/{portfolio}` - Risk analysis

#### Portfolio Management
- `GET /api/portfolio` - Get portfolio holdings
- `POST /api/portfolio/optimize` - Optimize allocation
- `POST /api/portfolio/rebalance` - Rebalancing suggestions
- `GET /api/portfolio/risk` - Risk metrics

### WebSocket Events

#### Client to Server
```javascript
// Subscribe to real-time updates
socket.emit('subscribe', {
    symbols: ['AAPL', 'GOOGL', 'TSLA'],
    room: 'dashboard'
});

// Unsubscribe from updates
socket.emit('unsubscribe', {
    room: 'dashboard'
});
```

#### Server to Client
```javascript
// Real-time market data
socket.on('market_data', (data) => {
    // Handle price updates
    console.log(data);
});

// AI prediction updates
socket.on('prediction_update', (data) => {
    // Handle ML predictions
    console.log(data.prediction);
});

// Trading alerts
socket.on('trading_alert', (alert) => {
    // Handle alerts
    console.log(alert.message);
});
```

## 🧠 Machine Learning Models

### LSTM Neural Network
- **Purpose**: Advanced stock price prediction
- **Architecture**: 3-layer LSTM with dropout regularization
- **Features**: OHLCV data + technical indicators
- **Fallback**: Linear Regression when TensorFlow unavailable

### Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA

### Portfolio Optimization
- **Method**: Modern Portfolio Theory
- **Optimization**: Sharpe ratio, minimum variance, maximum return
- **Risk Metrics**: VaR, Expected Shortfall, Maximum Drawdown

## 🛡️ Security

### Security Features
- **Environment-based secrets** management
- **SQL injection prevention** with parameterized queries
- **XSS protection** with proper input sanitization
- **CORS configuration** for cross-origin requests
- **Rate limiting** for API endpoints
- **Docker security** with non-root users

### Security Scanning
```bash
# Run security scans
bandit -r . -x tests/
safety check
docker scout cves
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale the application
docker-compose up -d --scale stock-ai-app=3

# Update deployment
docker-compose pull
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods,services,ingress

# View logs
kubectl logs -f deployment/stock-ai
```

### Cloud Deployment
- **AWS**: ECS, EKS, or Elastic Beanstalk
- **Google Cloud**: Cloud Run, GKE
- **Azure**: Container Instances, AKS
- **DigitalOcean**: App Platform, Kubernetes

## 📊 Performance

### Benchmarks
- **API Response Time**: < 200ms (cached), < 2s (uncached)
- **WebSocket Latency**: < 50ms
- **ML Prediction Time**: < 5s per symbol
- **Memory Usage**: ~512MB base, ~1GB with ML models
- **Database Connections**: 5-20 concurrent connections

### Optimization
- **Redis caching** for frequent queries
- **Database indexing** for fast lookups
- **Connection pooling** for database efficiency
- **Async processing** for non-blocking operations
- **WebSocket streaming** for real-time updates

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Database and API testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

## 📈 Monitoring

### Health Checks
- **Application**: `/api/status` endpoint
- **Database**: Connection pool status
- **Cache**: Redis connectivity
- **External APIs**: Rate limit monitoring

### Metrics
- **System**: CPU, Memory, Disk usage
- **Application**: Request rates, response times
- **Business**: Active users, predictions made
- **Errors**: Exception rates, failed requests

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/stock_ai.git
cd stock_ai

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Make your changes and commit
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### Code Standards
- **Python**: Follow PEP 8 style guide
- **Type Hints**: Use type annotations
- **Documentation**: Docstrings for all functions
- **Testing**: Write tests for new features
- **Security**: Follow security best practices

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- **API Docs**: `/docs` endpoint when running
- **Architecture**: See `docs/architecture.md`
- **Deployment**: See `docs/deployment.md`

### Getting Help
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Security**: Email security@stockai.com for vulnerabilities

### Community
- **Discord**: Join our developer community
- **Twitter**: Follow @StockAI for updates
- **Blog**: Read technical deep-dives

## 🎯 Roadmap

### Phase 1 (Current) ✅
- Core ML models and predictions
- Real-time streaming
- Portfolio optimization
- Risk management
- Docker deployment

### Phase 2 (Next)
- Mobile app development
- Advanced charting
- Options trading analysis
- Crypto currency support
- Social trading features

### Phase 3 (Future)
- Quantitative trading strategies
- Alternative data sources
- Institutional features
- Regulatory compliance
- Global market expansion

---

**Built with ❤️ for the fintech community**