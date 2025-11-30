import json
import time
from typing import Any, Dict, Optional

try:
    import redis  # type: ignore

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheManager:
    """Cache manager with Redis and in-memory fallback"""

    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Basic stats expected by tests
        self.stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "get_operations": 0,
            "set_operations": 0,
            "delete_operations": 0,
        }

    async def initialize(self):
        """Initialize cache system"""
        if REDIS_AVAILABLE:
            try:
                from config import config  # type: ignore

                self.redis_client = redis.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    password=config.redis.password,
                    db=config.redis.db,
                    decode_responses=True,
                )
                # Test connection
                self.redis_client.ping()
                print("✅ Redis cache initialized")
                return
            except Exception as e:
                print(f"⚠️ Redis unavailable, using memory cache: {e}")

        print("✅ In-memory cache initialized")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats["get_operations"] += 1

        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value is not None:
                    self.stats["hits"] += 1
                    return json.loads(value)
                else:
                    self.stats["misses"] += 1
            except Exception as e:
                print(f"Redis get error: {e}")

        # Fallback to memory cache
        item = self.memory_cache.get(key)
        if item:
            if time.time() - item["timestamp"] < item.get("ttl", self.cache_ttl):
                self.stats["hits"] += 1
                return item["value"]
            else:
                # expired
                del self.memory_cache[key]
        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        self.stats["set_operations"] += 1
        ttl = ttl or self.cache_ttl

        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                print(f"Redis set error: {e}")

        # Fallback to memory cache
        self.memory_cache[key] = {"value": value, "timestamp": time.time(), "ttl": ttl}

        # Clean old entries periodically
        if len(self.memory_cache) > 1000:
            self._cleanup_memory_cache()

        return True

    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        self.stats["delete_operations"] += 1

        # Try Redis first
        if self.redis_client:
            try:
                deleted = self.redis_client.delete(key)
                return bool(deleted)
            except Exception as e:
                print(f"Redis delete error: {e}")

        # Fallback to memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            return True
        return False

    async def cache_stock_data(
        self, ticker: str, timeframe: str, data: Dict[str, Any], ttl: int = 300
    ) -> bool:
        """Helper to cache stock data by ticker and timeframe"""
        key = f"stock:{ticker}:{timeframe}"
        return await self.set(key, data, ttl)

    async def get_stock_data(
        self, ticker: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Helper to retrieve stock data by ticker and timeframe"""
        key = f"stock:{ticker}:{timeframe}"
        value = await self.get(key)
        if isinstance(value, dict):
            return value
        return None

    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        current_time = time.time()
        # Use list comprehension for efficient key collection
        keys_to_delete = [
            key
            for key, item in self.memory_cache.items()
            if current_time - item["timestamp"] > item.get("ttl", self.cache_ttl)
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hits = int(self.stats["hits"])
        misses = int(self.stats["misses"])
        total_lookups = hits + misses
        hit_rate_percent = (hits / total_lookups) * 100.0 if total_lookups > 0 else 0.0

        return {
            "backend": "Redis" if self.redis_client else "Memory",
            "memory_cache_size": len(self.memory_cache),
            "hits": hits,
            "misses": misses,
            "hit_rate_percent": round(hit_rate_percent, 2),
            "get_operations": int(self.stats["get_operations"]),
            "set_operations": int(self.stats["set_operations"]),
            "delete_operations": int(self.stats["delete_operations"]),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        status = {"status": "healthy"}
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["redis"] = "connected"
            except Exception:
                status["redis"] = "disconnected"
        status["memory_cache"] = f"{len(self.memory_cache)} items"
        return status

    def get_technical_indicators(
        self, ticker: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached technical indicators (stub for compatibility)"""
        return None


# Global cache manager instance
cache_manager = CacheManager()
