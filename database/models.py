"""
Database models for the enterprise fintech platform
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TimeFrame(Enum):
    """Supported timeframes for analysis"""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class StockData:
    """Stock price and volume data model"""

    id: Optional[int] = None
    symbol: str = ""
    timestamp: datetime = None
    timeframe: TimeFrame = TimeFrame.DAY_1
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    created_at: datetime = None


@dataclass
class TechnicalIndicator:
    """Technical indicator values"""

    id: Optional[int] = None
    symbol: str = ""
    timestamp: datetime = None
    timeframe: TimeFrame = TimeFrame.DAY_1

    # Moving averages
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # Momentum indicators
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None

    # Volume indicators
    volume_sma: Optional[float] = None
    on_balance_volume: Optional[float] = None

    created_at: datetime = None


@dataclass
class PredictionHistory:
    """ML model prediction history"""

    id: Optional[int] = None
    symbol: str = ""
    model_type: str = "LSTM"
    prediction_date: datetime = None
    target_date: datetime = None
    predicted_price: float = 0.0
    actual_price: Optional[float] = None
    confidence_score: float = 0.0
    features_used: Optional[Dict[str, Any]] = None
    accuracy: Optional[float] = None
    created_at: datetime = None


@dataclass
class UserProfile:
    """User profile and preferences"""

    id: Optional[int] = None
    username: str = ""
    email: Optional[str] = None
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    investment_horizon: str = "medium"  # short, medium, long
    initial_capital: float = 10000.0
    preferred_sectors: Optional[list] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class Portfolio:
    """User portfolio holdings"""

    id: Optional[int] = None
    user_id: int = 0
    symbol: str = ""
    shares: float = 0.0
    avg_purchase_price: float = 0.0
    current_price: float = 0.0
    purchase_date: datetime = None
    last_updated: datetime = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_gain_loss(self) -> float:
        return (self.current_price - self.avg_purchase_price) * self.shares

    @property
    def percentage_change(self) -> float:
        if self.avg_purchase_price == 0:
            return 0.0
        return (
            (self.current_price - self.avg_purchase_price) / self.avg_purchase_price
        ) * 100
