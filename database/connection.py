"""
Database connection and management for the enterprise platform
"""

import asyncio
import asyncpg
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from config import config
from .models import (
    StockData,
    TechnicalIndicator,
    PredictionHistory,
    UserProfile,
    Portfolio,
    TimeFrame,
)


class DatabaseManager:
    """Enterprise-grade database manager with connection pooling"""

    def __init__(self):
        self.pool = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection pool"""
        if self._initialized:
            return

        try:
            self.pool = await asyncpg.create_pool(
                config.database.connection_string,
                min_size=config.database.pool_min_size,
                max_size=config.database.pool_max_size,
                command_timeout=60,
            )
            await self._create_tables()
            self._initialized = True
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            # Don't raise exception to allow fallback operation
            self._initialized = False

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            self._initialized = False

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized or not self.pool:
            raise Exception("Database not available")

        async with self.pool.acquire() as connection:
            yield connection

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self.get_connection() as conn:
            # Stock data table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    timeframe VARCHAR(5) NOT NULL,
                    open_price DECIMAL(10,2) NOT NULL,
                    high_price DECIMAL(10,2) NOT NULL,
                    low_price DECIMAL(10,2) NOT NULL,
                    close_price DECIMAL(10,2) NOT NULL,
                    volume BIGINT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(symbol, timestamp, timeframe)
                );
            """
            )

            # Technical indicators table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    timeframe VARCHAR(5) NOT NULL,
                    sma_10 DECIMAL(10,2),
                    sma_20 DECIMAL(10,2),
                    sma_50 DECIMAL(10,2),
                    ema_12 DECIMAL(10,2),
                    ema_26 DECIMAL(10,2),
                    rsi DECIMAL(5,2),
                    macd_line DECIMAL(10,4),
                    macd_signal DECIMAL(10,4),
                    macd_histogram DECIMAL(10,4),
                    bollinger_upper DECIMAL(10,2),
                    bollinger_middle DECIMAL(10,2),
                    bollinger_lower DECIMAL(10,2),
                    atr DECIMAL(10,4),
                    volume_sma BIGINT,
                    on_balance_volume BIGINT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(symbol, timestamp, timeframe)
                );
            """
            )

            # Prediction history table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    model_type VARCHAR(20) NOT NULL,
                    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    target_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    predicted_price DECIMAL(10,2) NOT NULL,
                    actual_price DECIMAL(10,2),
                    confidence_score DECIMAL(5,2) NOT NULL,
                    features_used JSONB,
                    accuracy DECIMAL(5,2),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            )

            # User profiles table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE,
                    risk_tolerance VARCHAR(20) DEFAULT 'moderate',
                    investment_horizon VARCHAR(20) DEFAULT 'medium',
                    initial_capital DECIMAL(15,2) DEFAULT 10000.00,
                    preferred_sectors JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            )

            # Portfolio table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES user_profiles(id),
                    symbol VARCHAR(10) NOT NULL,
                    shares DECIMAL(15,6) NOT NULL,
                    avg_purchase_price DECIMAL(10,2) NOT NULL,
                    current_price DECIMAL(10,2),
                    purchase_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(user_id, symbol)
                );
            """
            )

            # Create indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timestamp ON stock_data(symbol, timestamp);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prediction_history_symbol_date ON prediction_history(symbol, prediction_date);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol ON portfolio(user_id, symbol);"
            )

    async def insert_stock_data(self, data: StockData):
        """Insert stock price data"""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO stock_data 
                (symbol, timestamp, timeframe, open_price, high_price, low_price, close_price, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume
            """,
                data.symbol,
                data.timestamp,
                data.timeframe.value,
                data.open_price,
                data.high_price,
                data.low_price,
                data.close_price,
                data.volume,
            )


# Global database manager instance
db_manager = DatabaseManager()
