# Deployment Guide for Stock AI Platform

## Render Deployment

### Quick Deploy to Render

1. **Fork or Clone Repository**
   ```bash
   git clone https://github.com/Anamitra-Sarkar/stock_ai.git
   cd stock_ai
   ```

2. **Environment Variables**
   - Copy `.env.example` to `.env` 
   - Set deployment-optimized variables:
   ```bash
   DEBUG=false
   MINIMAL_STARTUP=true
   SKIP_ML_TRAINING=true
   ```

3. **Create Render Web Service**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`
   - Environment: `Python 3.11+`

### Required Environment Variables for Render

| Variable | Value | Required | Description |
|----------|-------|----------|-------------|
| `MINIMAL_STARTUP` | `true` | Yes | Enables fast startup for deployment |
| `SKIP_ML_TRAINING` | `true` | Yes | Skips expensive ML training on startup |
| `HOST` | `0.0.0.0` | Yes | Required for Render |
| `PORT` | `5000` | No | Auto-set by Render |
| `SECRET_KEY` | Your secret key | Yes | Flask secret key |

### Optional Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHA_VANTAGE_API_KEY` | Stock data API | Mock data used if missing |
| `REDIS_HOST` | Redis server | Falls back to in-memory cache |
| `DB_HOST` | PostgreSQL host | Falls back to mock data |

### Health Check Endpoint

The application provides health endpoints for monitoring:
- `GET /health` - Basic health status  
- `GET /api/health` - Detailed health with service status

### Features in Deployment Mode

✅ **Available:**
- REST API endpoints
- WebSocket streaming
- Portfolio optimization  
- Risk analysis
- Technical indicators
- Sentiment analysis
- In-memory caching
- Mock data generation

⚠️ **Limited:**
- ML model training (disabled for fast startup)
- Database persistence (optional)
- Redis caching (falls back to memory)

### Troubleshooting

**Common Issues:**

1. **Slow Startup**
   - Ensure `SKIP_ML_TRAINING=true` is set
   - Set `MINIMAL_STARTUP=true`

2. **Memory Issues**
   - App uses in-memory fallbacks when external services unavailable
   - Restart if memory usage grows excessive

3. **API Rate Limiting**
   - Mock data is used when no API key provided
   - Set `ALPHA_VANTAGE_API_KEY` for real data

### Production Scaling

For production environments:

1. **Enable External Services**
   ```bash
   REDIS_HOST=your-redis-host
   DB_HOST=your-postgres-host
   ALPHA_VANTAGE_API_KEY=your-key
   ```

2. **Enable ML Training** (only if you have sufficient compute)
   ```bash
   SKIP_ML_TRAINING=false
   LSTM_EPOCHS=20
   ```

3. **Security**
   ```bash
   DEBUG=false
   SECRET_KEY=your-strong-secret-key
   ```

### API Endpoints

Once deployed, your API will be available at:

- `GET /` - Frontend interface
- `GET /health` - Health check
- `GET /api/status` - Detailed system status
- `GET /api/dashboard` - Market dashboard
- `POST /api/ask` - AI advisor queries
- `GET /api/prediction/<ticker>` - Stock predictions
- `POST /api/portfolio/optimize` - Portfolio optimization

### Example Render Configuration

```yaml
# render.yaml
services:
- type: web
  name: stock-ai
  env: python
  buildCommand: pip install -r requirements.txt
  startCommand: python main.py
  envVars:
  - key: MINIMAL_STARTUP
    value: "true"
  - key: SKIP_ML_TRAINING  
    value: "true"
  - key: SECRET_KEY
    generateValue: true
```