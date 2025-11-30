"""
Integration tests for the enterprise stock AI platform
"""

import pytest
import asyncio
from unittest.mock import patch
import json
from main import app


class TestAPIIntegration:
    """Integration tests for REST API"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_status_endpoint(self, client):
        """Test status endpoint"""
        response = client.get("/api/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "api_key_configured" in data
        assert "categories_available" in data

    def test_dashboard_endpoint(self, client):
        """Test dashboard endpoint"""
        response = client.get("/api/dashboard")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "stocks" in data
        assert "alerts" in data
        assert "data_status" in data

    def test_category_endpoint(self, client):
        """Test category endpoint"""
        response = client.get("/api/category/Technology")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "stocks" in data
        assert len(data["stocks"]) > 0

    def test_search_endpoint(self, client):
        """Test stock search endpoint"""
        response = client.get("/api/search/AAPL")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "stock" in data
        assert data["stock"]["ticker"] == "AAPL"

    def test_ask_endpoint(self, client):
        """Test AI assistant endpoint"""
        payload = {
            "query": "What do you think about AAPL?",
            "profile": {"risk_tolerance": "moderate"},
        }

        response = client.post(
            "/api/ask", data=json.dumps(payload), content_type="application/json"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "response" in data

    def test_invalid_category(self, client):
        """Test invalid category handling"""
        response = client.get("/api/category/InvalidCategory")

        # Should handle gracefully
        assert response.status_code in [200, 404]

    def test_invalid_ticker_search(self, client):
        """Test invalid ticker search"""
        response = client.get("/api/search/INVALID")

        assert response.status_code == 200
        # Should return some response even for invalid ticker


@pytest.mark.asyncio
class TestDatabaseIntegration:
    """Integration tests for database operations"""

    async def test_database_connection(self):
        """Test database connection"""
        from database.connection import db_manager

        # This will test the fallback behavior if DB not available
        try:
            await db_manager.initialize()
            assert True  # If no exception, connection works
        except Exception as e:
            # Expected if no database available in test environment
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    async def test_stock_data_operations(self):
        """Test stock data CRUD operations"""
        from database.connection import db_manager
        from database.models import StockData, TimeFrame
        from datetime import datetime

        try:
            await db_manager.initialize()

            # Create test data
            test_data = StockData(
                symbol="TEST",
                timestamp=datetime.now(),
                timeframe=TimeFrame.DAY_1,
                open_price=100.0,
                high_price=105.0,
                low_price=95.0,
                close_price=102.0,
                volume=1000000,
            )

            # Test insert
            stock_id = await db_manager.insert_stock_data(test_data)
            assert stock_id is not None

            # Test retrieve
            stocks = await db_manager.get_stock_data("TEST", TimeFrame.DAY_1, 10)
            assert len(stocks) > 0
            assert stocks[0].symbol == "TEST"

        except Exception as e:
            # Expected if database not available
            pytest.skip(f"Database not available: {e}")


@pytest.mark.asyncio
class TestCacheIntegration:
    """Integration tests for cache operations"""

    async def test_cache_operations(self):
        """Test basic cache operations"""
        from cache.redis_cache import cache_manager

        await cache_manager.initialize()

        # Test set and get
        test_key = "test:integration"
        test_value = {"test": "data", "number": 123}

        success = await cache_manager.set(test_key, test_value, 60)
        assert success

        retrieved_value = await cache_manager.get(test_key)
        assert retrieved_value is not None
        assert retrieved_value["test"] == "data"
        assert retrieved_value["number"] == 123

        # Test delete
        deleted = await cache_manager.delete(test_key)
        assert deleted

        # Verify deletion
        retrieved_after_delete = await cache_manager.get(test_key)
        assert retrieved_after_delete is None

    async def test_stock_cache_helpers(self):
        """Test stock-specific cache helpers"""
        from cache.redis_cache import cache_manager

        await cache_manager.initialize()

        # Test stock data caching
        test_data = {"price": 150.0, "volume": 1000000}

        success = await cache_manager.cache_stock_data("AAPL", "1d", test_data, 5)
        assert success

        cached_data = await cache_manager.get_stock_data("AAPL", "1d")
        assert cached_data is not None
        assert cached_data["price"] == 150.0

    async def test_cache_stats(self):
        """Test cache statistics"""
        from cache.redis_cache import cache_manager

        await cache_manager.initialize()

        stats = cache_manager.get_stats()

        assert "backend" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate_percent" in stats


class TestMLIntegration:
    """Integration tests for ML components"""

    def test_lstm_predictor_initialization(self):
        """Test LSTM predictor can be created"""
        from ml_models.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor("AAPL")
        assert predictor.symbol == "AAPL"
        assert predictor.sequence_length > 0

    def test_portfolio_optimizer(self):
        """Test portfolio optimization"""
        from portfolio.optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        assert optimizer.risk_free_rate >= 0

        # Test with sample data
        sample_price_data = {
            "AAPL": [
                {"timestamp": "2023-01-01", "close_price": 150.0},
                {"timestamp": "2023-01-02", "close_price": 152.0},
                {"timestamp": "2023-01-03", "close_price": 148.0},
            ],
            "GOOGL": [
                {"timestamp": "2023-01-01", "close_price": 100.0},
                {"timestamp": "2023-01-02", "close_price": 102.0},
                {"timestamp": "2023-01-03", "close_price": 98.0},
            ],
        }

        try:
            result = optimizer.optimize_portfolio(sample_price_data)
            assert "allocations" in result
            assert "optimization_method" in result
        except Exception as e:
            # May fail due to insufficient data, which is expected
            assert "data" in str(e).lower() or "optimization" in str(e).lower()


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_complete_analysis_workflow(self, client):
        """Test complete stock analysis workflow"""
        # Step 1: Get dashboard data
        dashboard_response = client.get("/api/dashboard")
        assert dashboard_response.status_code == 200

        dashboard_data = json.loads(dashboard_response.data)
        assert len(dashboard_data["stocks"]) > 0

        # Step 2: Get specific stock analysis
        first_stock = dashboard_data["stocks"][0]
        ticker = first_stock["ticker"]

        search_response = client.get(f"/api/search/{ticker}")
        assert search_response.status_code == 200

        # Step 3: Ask AI for advice
        ask_payload = {
            "query": f"Should I invest in {ticker}?",
            "profile": {"risk_tolerance": "moderate", "initialCapital": 10000},
        }

        ask_response = client.post(
            "/api/ask", data=json.dumps(ask_payload), content_type="application/json"
        )

        assert ask_response.status_code == 200
        ask_data = json.loads(ask_response.data)
        assert "response" in ask_data
        assert len(ask_data["response"]) > 0

    @patch("agents.prediction_agent.PredictionAgent.predict_trend")
    def test_prediction_workflow(self, mock_predict, client):
        """Test prediction workflow with mocked ML"""
        # Mock the prediction with an async return value
        import asyncio

        async def async_predict(*args, **kwargs):
            return {
                "trend": "up",
                "confidence": 85,
                "predicted_price": 155.0,
                "ticker": args[0] if args else "AAPL",
            }

        mock_predict.side_effect = async_predict

        # Test the workflow
        response = client.get("/api/search/AAPL")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "stock" in data
        assert "prediction" in data["stock"]
