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
from .models import StockData, TechnicalIndicator, PredictionHistory, UserProfile, Portfolio, TimeFrame

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
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            await self._create_tables()
            self._initialized = True
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise
    
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
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self.get_connection() as conn:
            # Stock data table
            await conn.execute("""
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
            """)
            
            # Technical indicators table
            await conn.execute("""
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
            """)
            
            # Prediction history table
            await conn.execute("""
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
            """)
            
            # User profiles table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100),
                    risk_tolerance VARCHAR(20) DEFAULT 'moderate',
                    investment_horizon VARCHAR(20) DEFAULT 'medium',
                    initial_capital DECIMAL(15,2) DEFAULT 10000.00,
                    preferred_sectors JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Portfolio table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_profiles(id),
                    symbol VARCHAR(10) NOT NULL,
                    shares DECIMAL(15,6) NOT NULL,
                    avg_purchase_price DECIMAL(10,2) NOT NULL,
                    current_price DECIMAL(10,2) DEFAULT 0.00,
                    purchase_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(user_id, symbol)
                );
            """)
            
            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_timestamp ON stock_data(symbol, timestamp);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_history_symbol_date ON prediction_history(symbol, prediction_date);")
    
    # Stock Data Operations
    async def insert_stock_data(self, stock_data: StockData) -> int:
        """Insert stock data record"""
        async with self.get_connection() as conn:
            result = await conn.fetchval("""
                INSERT INTO stock_data 
                (symbol, timestamp, timeframe, open_price, high_price, low_price, close_price, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, timestamp, timeframe) 
                DO UPDATE SET 
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
                RETURNING id;
            """, stock_data.symbol, stock_data.timestamp, stock_data.timeframe.value,
                stock_data.open_price, stock_data.high_price, stock_data.low_price,
                stock_data.close_price, stock_data.volume)
            return result
    
    async def get_stock_data(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[StockData]:
        """Get historical stock data"""
        async with self.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM stock_data 
                WHERE symbol = $1 AND timeframe = $2 
                ORDER BY timestamp DESC 
                LIMIT $3;
            """, symbol, timeframe.value, limit)
            
            return [StockData(
                id=row['id'],
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                timeframe=TimeFrame(row['timeframe']),
                open_price=float(row['open_price']),
                high_price=float(row['high_price']),
                low_price=float(row['low_price']),
                close_price=float(row['close_price']),
                volume=row['volume'],
                created_at=row['created_at']
            ) for row in rows]
    
    # Technical Indicators Operations
    async def insert_technical_indicators(self, indicator: TechnicalIndicator) -> int:
        """Insert technical indicators"""
        async with self.get_connection() as conn:
            result = await conn.fetchval("""
                INSERT INTO technical_indicators 
                (symbol, timestamp, timeframe, sma_10, sma_20, sma_50, ema_12, ema_26, 
                 rsi, macd_line, macd_signal, macd_histogram, bollinger_upper, 
                 bollinger_middle, bollinger_lower, atr, volume_sma, on_balance_volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (symbol, timestamp, timeframe)
                DO UPDATE SET 
                    sma_10 = EXCLUDED.sma_10, sma_20 = EXCLUDED.sma_20, sma_50 = EXCLUDED.sma_50,
                    ema_12 = EXCLUDED.ema_12, ema_26 = EXCLUDED.ema_26, rsi = EXCLUDED.rsi,
                    macd_line = EXCLUDED.macd_line, macd_signal = EXCLUDED.macd_signal,
                    macd_histogram = EXCLUDED.macd_histogram, bollinger_upper = EXCLUDED.bollinger_upper,
                    bollinger_middle = EXCLUDED.bollinger_middle, bollinger_lower = EXCLUDED.bollinger_lower,
                    atr = EXCLUDED.atr, volume_sma = EXCLUDED.volume_sma, 
                    on_balance_volume = EXCLUDED.on_balance_volume
                RETURNING id;
            """, indicator.symbol, indicator.timestamp, indicator.timeframe.value,
                indicator.sma_10, indicator.sma_20, indicator.sma_50, indicator.ema_12,
                indicator.ema_26, indicator.rsi, indicator.macd_line, indicator.macd_signal,
                indicator.macd_histogram, indicator.bollinger_upper, indicator.bollinger_middle,
                indicator.bollinger_lower, indicator.atr, indicator.volume_sma, 
                indicator.on_balance_volume)
            return result
    
    # Prediction Operations
    async def insert_prediction(self, prediction: PredictionHistory) -> int:
        """Insert prediction record"""
        async with self.get_connection() as conn:
            result = await conn.fetchval("""
                INSERT INTO prediction_history 
                (symbol, model_type, prediction_date, target_date, predicted_price, 
                 actual_price, confidence_score, features_used, accuracy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id;
            """, prediction.symbol, prediction.model_type, prediction.prediction_date,
                prediction.target_date, prediction.predicted_price, prediction.actual_price,
                prediction.confidence_score, json.dumps(prediction.features_used),
                prediction.accuracy)
            return result

# Global database manager instance
db_manager = DatabaseManager()