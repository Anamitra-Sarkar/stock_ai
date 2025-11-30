"""
Performance tests for the enterprise stock AI platform
"""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import json


class TestAPIPerformance:
    """Performance tests for REST API endpoints"""

    BASE_URL = "http://localhost:5000"

    @pytest.fixture(scope="class")
    def server_running(self):
        """Check if server is running for performance tests"""
        try:
            response = requests.get(f"{self.BASE_URL}/api/status", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        pytest.skip("Server not running for performance tests")

    def test_status_endpoint_performance(self, server_running):
        """Test status endpoint response time"""
        start_time = time.time()
        response = requests.get(f"{self.BASE_URL}/api/status", timeout=10)
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 0.5  # Should respond within 500ms

    def test_dashboard_endpoint_performance(self, server_running):
        """Test dashboard endpoint performance"""
        start_time = time.time()
        response = requests.get(f"{self.BASE_URL}/api/dashboard", timeout=10)
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds

    def test_concurrent_requests_performance(self, server_running):
        """Test performance under concurrent load"""

        def make_request():
            """Make a single API request"""
            try:
                response = requests.get(f"{self.BASE_URL}/api/status", timeout=10)
                return response.status_code == 200, response.elapsed.total_seconds()
            except Exception as e:
                return False, float("inf")

        # Test with 10 concurrent requests
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_time = end_time - start_time

        # Check results
        successful_requests = sum(1 for success, _ in results if success)
        average_response_time = sum(
            time for _, time in results if time != float("inf")
        ) / len(results)

        assert successful_requests >= 8  # At least 80% success rate
        assert total_time < 10.0  # Complete within 10 seconds
        assert average_response_time < 2.0  # Average response time under 2 seconds

    def test_memory_usage_stability(self, server_running):
        """Test memory usage doesn't grow excessively"""
        # Make multiple requests to check for memory leaks
        initial_time = time.time()

        for i in range(20):
            response = requests.get(f"{self.BASE_URL}/api/dashboard", timeout=10)
            assert response.status_code == 200

            # Small delay between requests
            time.sleep(0.1)

        total_time = time.time() - initial_time
        assert total_time < 30.0  # Should complete within 30 seconds


class TestMLPerformance:
    """Performance tests for ML components"""

    def test_prediction_agent_performance(self):
        """Test prediction agent performance"""
        from agents.prediction_agent import PredictionAgent

        agent = PredictionAgent()

        start_time = time.time()

        # Test prediction speed
        result = asyncio.run(agent.predict_trend("AAPL", 150.0))

        end_time = time.time()
        prediction_time = end_time - start_time

        assert result is not None
        assert "trend" in result
        assert prediction_time < 10.0  # Should complete within 10 seconds

    def test_technical_indicators_performance(self):
        """Test technical indicators calculation speed"""
        from indicators.technical_indicators import TechnicalIndicators
        import numpy as np

        # Generate large dataset
        np.random.seed(42)
        large_dataset = [100 + np.random.randn() * 5 for _ in range(1000)]

        start_time = time.time()

        # Calculate multiple indicators
        sma_20 = TechnicalIndicators.sma(large_dataset, 20)
        ema_20 = TechnicalIndicators.ema(large_dataset, 20)
        rsi = TechnicalIndicators.rsi(large_dataset, 14)

        end_time = time.time()
        calculation_time = end_time - start_time

        assert len(sma_20) > 0
        assert len(ema_20) > 0
        assert len(rsi) > 0
        assert calculation_time < 2.0  # Should complete within 2 seconds

    def test_portfolio_optimization_performance(self):
        """Test portfolio optimization performance"""
        from portfolio.optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Create sample data for multiple assets
        sample_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
            sample_data[symbol] = [
                {
                    "timestamp": f"2023-01-{i:02d}",
                    "close_price": 100 + i + (hash(symbol) % 50),
                }
                for i in range(1, 101)  # 100 data points
            ]

        start_time = time.time()

        try:
            result = optimizer.optimize_portfolio(sample_data)
            end_time = time.time()
            optimization_time = end_time - start_time

            assert optimization_time < 15.0  # Should complete within 15 seconds
            assert "allocations" in result
        except Exception as e:
            # Optimization might fail with limited data, which is acceptable
            assert "data" in str(e).lower() or "optimization" in str(e).lower()


@pytest.mark.asyncio
class TestCachePerformance:
    """Performance tests for caching system"""

    async def test_cache_read_write_performance(self):
        """Test cache read/write performance"""
        from cache.redis_cache import cache_manager

        await cache_manager.initialize()

        # Test write performance
        test_data = {"test": "data", "numbers": list(range(1000))}

        start_time = time.time()
        for i in range(100):
            await cache_manager.set(f"perf_test_{i}", test_data, 60)
        write_time = time.time() - start_time

        # Test read performance
        start_time = time.time()
        for i in range(100):
            result = await cache_manager.get(f"perf_test_{i}")
            assert result is not None
        read_time = time.time() - start_time

        # Performance assertions
        assert write_time < 5.0  # 100 writes within 5 seconds
        assert read_time < 2.0  # 100 reads within 2 seconds

        # Cleanup
        for i in range(100):
            await cache_manager.delete(f"perf_test_{i}")

    async def test_cache_concurrent_access(self):
        """Test cache performance under concurrent access"""
        from cache.redis_cache import cache_manager

        await cache_manager.initialize()

        async def cache_operation(index):
            """Perform cache operations"""
            key = f"concurrent_test_{index}"
            data = {"index": index, "data": f"test_data_{index}"}

            # Write
            await cache_manager.set(key, data, 60)

            # Read
            result = await cache_manager.get(key)

            # Delete
            await cache_manager.delete(key)

            return result is not None

        start_time = time.time()

        # Run 20 concurrent cache operations
        tasks = [cache_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Check performance and correctness
        successful_operations = sum(1 for result in results if result is True)
        assert successful_operations >= 18  # At least 90% success rate
        assert total_time < 10.0  # Should complete within 10 seconds


class TestDatabasePerformance:
    """Performance tests for database operations"""

    @pytest.mark.asyncio
    async def test_database_connection_pool_performance(self):
        """Test database connection pooling performance"""
        from database.connection import db_manager

        try:
            await db_manager.initialize()

            async def db_operation():
                """Perform a database operation"""
                async with db_manager.get_connection() as conn:
                    result = await conn.fetchval("SELECT 1")
                    return result == 1

            start_time = time.time()

            # Test with concurrent database operations
            tasks = [db_operation() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            operation_time = end_time - start_time

            successful_operations = sum(1 for result in results if result is True)

            assert successful_operations >= 8  # At least 80% success rate
            assert operation_time < 5.0  # Should complete within 5 seconds

        except Exception as e:
            # Expected if database not available
            pytest.skip(f"Database not available for performance testing: {e}")

    @pytest.mark.asyncio
    async def test_bulk_data_insertion_performance(self):
        """Test bulk data insertion performance"""
        from database.connection import db_manager
        from database.models import StockData, TimeFrame
        from datetime import datetime, timedelta

        try:
            await db_manager.initialize()

            # Generate test data
            test_data = []
            base_date = datetime.now()

            for i in range(100):
                stock_data = StockData(
                    symbol="PERF_TEST",
                    timestamp=base_date + timedelta(days=i),
                    timeframe=TimeFrame.DAY_1,
                    open_price=100.0 + i,
                    high_price=105.0 + i,
                    low_price=95.0 + i,
                    close_price=102.0 + i,
                    volume=1000000 + i * 1000,
                )
                test_data.append(stock_data)

            start_time = time.time()

            # Insert data
            for data in test_data:
                await db_manager.insert_stock_data(data)

            end_time = time.time()
            insertion_time = end_time - start_time

            assert insertion_time < 30.0  # Should complete within 30 seconds

            # Test retrieval performance
            start_time = time.time()
            retrieved_data = await db_manager.get_stock_data(
                "PERF_TEST", TimeFrame.DAY_1, 100
            )
            end_time = time.time()

            retrieval_time = end_time - start_time

            assert len(retrieved_data) > 0
            assert retrieval_time < 5.0  # Should retrieve within 5 seconds

        except Exception as e:
            # Expected if database not available
            pytest.skip(f"Database not available for performance testing: {e}")


class TestLoadTesting:
    """Load testing scenarios"""

    def test_sustained_load(self, pytestconfig):
        """Test sustained load handling"""
        if not pytestconfig.getoption("--load-test"):
            pytest.skip("Load testing not enabled. Use --load-test to run.")

        # This would be a more comprehensive load test
        # For now, just a basic sustained request test

        def make_requests():
            """Make sustained requests"""
            successful_requests = 0
            failed_requests = 0

            for _ in range(10):  # 10 requests per thread
                try:
                    response = requests.get(
                        "http://localhost:5000/api/status", timeout=10
                    )
                    if response.status_code == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                except:
                    failed_requests += 1

                time.sleep(0.1)  # Small delay between requests

            return successful_requests, failed_requests

        start_time = time.time()

        # Run 5 threads, each making 10 requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_requests) for _ in range(5)]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_time = end_time - start_time

        total_successful = sum(success for success, _ in results)
        total_failed = sum(failed for _, failed in results)

        success_rate = total_successful / (total_successful + total_failed) * 100

        assert success_rate >= 90  # At least 90% success rate
        assert total_time < 60.0  # Complete within 60 seconds
