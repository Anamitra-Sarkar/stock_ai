"""
Configuration management for the enterprise fintech platform
"""

import os
import secrets
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "stock_ai")
    username: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    pool_min_size: int = int(os.getenv("DB_POOL_MIN", 1))
    pool_max_size: int = int(os.getenv("DB_POOL_MAX", 5))

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration"""

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", 6379))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    db: int = int(os.getenv("REDIS_DB", 0))


@dataclass
class APIConfig:
    """API configuration"""

    alpha_vantage_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    finnhub_key: Optional[str] = os.getenv("FINNHUB_API_KEY")
    rate_limit_per_hour: int = int(os.getenv("API_RATE_LIMIT", 500))


@dataclass
class MLConfig:
    """Machine Learning configuration"""

    lstm_sequence_length: int = int(os.getenv("LSTM_SEQUENCE_LENGTH", 60))
    lstm_epochs: int = int(os.getenv("LSTM_EPOCHS", 50))
    lstm_batch_size: int = int(os.getenv("LSTM_BATCH_SIZE", 32))
    model_save_path: str = os.getenv("MODEL_SAVE_PATH", "./models")


class AppConfig:
    """Main application configuration"""

    def __init__(self):
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        # Default to localhost for security, but allow override via environment
        self.host = os.getenv("HOST", "127.0.0.1")  # More secure default
        self.port = int(os.getenv("PORT", 5000))
        # Generate a secure secret key if not provided
        self.secret_key = os.getenv("SECRET_KEY") or self._generate_secret_key()

        # Service configurations
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.ml = MLConfig()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key using cryptographically strong random bytes"""
        return secrets.token_hex(32)  # 256-bit key


# Global config instance
config = AppConfig()
