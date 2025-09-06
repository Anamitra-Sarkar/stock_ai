"""Redis cache with in-memory fallback"""
import json
import time
from typing import Any, Optional, Dict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheManager:
    """Cache manager with Redis and in-memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize cache system"""
        if REDIS_AVAILABLE:
            try:
                from config import config
                self.redis_client = redis.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    password=config.redis.password,
                    db=config.redis.db,
                    decode_responses=True
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
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            cached_item = self.memory_cache[key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['value']
            else:
                del self.memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.cache_ttl
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                print(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
        # Clean old entries periodically
        if len(self.memory_cache) > 1000:
            self._cleanup_memory_cache()
        
        return True
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        current_time = time.time()
        keys_to_delete = []
        
        for key, item in self.memory_cache.items():
            if current_time - item['timestamp'] > self.cache_ttl:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.memory_cache[key]
