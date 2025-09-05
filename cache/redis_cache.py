"""
Redis cache manager with in-memory fallback for deployment safety
"""
import json
import time
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available, using in-memory fallback")

from config import config

logger = logging.getLogger(__name__)

class CacheManager:
    """Enterprise cache manager with Redis and in-memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'backend': 'not_initialized'
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize Redis connection with fallback"""
        try:
            if REDIS_AVAILABLE:
                # Try to connect to Redis
                self.redis_client = redis.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    password=config.redis.password,
                    db=config.redis.db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                
                # Test connection
                self.redis_client.ping()
                self.cache_stats['backend'] = 'redis'
                print("✅ Redis cache initialized")
                
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
            
        # Always fall back to memory cache if Redis fails
        if self.redis_client is None:
            self.cache_stats['backend'] = 'memory'
            print("✅ In-memory cache fallback initialized")
            
        self._initialized = True
        
    def _serialize_key(self, key: str) -> str:
        """Ensure key is string"""
        return str(key)
        
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)
            
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from storage"""
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        key = self._serialize_key(key)
        
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value is not None:
                    self.cache_stats['hits'] += 1
                    return self._deserialize_value(value)
                else:
                    self.cache_stats['misses'] += 1
                    return None
            else:
                # Memory fallback
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    # Check expiration
                    if cache_entry['expires_at'] > time.time():
                        self.cache_stats['hits'] += 1
                        return cache_entry['value']
                    else:
                        del self.memory_cache[key]
                        
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None
            
    async def set(self, key: str, value: Any, expire_seconds: int = 3600) -> bool:
        """Set value in cache with expiration"""
        key = self._serialize_key(key)
        
        try:
            if self.redis_client:
                serialized_value = self._serialize_value(value)
                result = self.redis_client.setex(key, expire_seconds, serialized_value)
                return bool(result)
            else:
                # Memory fallback
                self.memory_cache[key] = {
                    'value': value,
                    'expires_at': time.time() + expire_seconds
                }
                return True
                
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        key = self._serialize_key(key)
        
        try:
            if self.redis_client:
                result = self.redis_client.delete(key)
                return bool(result)
            else:
                # Memory fallback
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        key = self._serialize_key(key)
        
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                # Memory fallback - check expiration
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry['expires_at'] > time.time():
                        return True
                    else:
                        del self.memory_cache[key]
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
            
    async def flush_all(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                result = self.redis_client.flushdb()
                return bool(result)
            else:
                # Memory fallback
                self.memory_cache.clear()
                return True
                
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        stats['total_requests'] = stats['hits'] + stats['misses']
        stats['hit_rate'] = (stats['hits'] / max(stats['total_requests'], 1)) * 100
        
        if self.cache_stats['backend'] == 'memory':
            stats['memory_entries'] = len(self.memory_cache)
            # Clean up expired entries for stats
            current_time = time.time()
            expired_count = sum(1 for entry in self.memory_cache.values() 
                              if entry['expires_at'] <= current_time)
            stats['expired_entries'] = expired_count
            
        return stats
        
    def _cleanup_memory_cache(self):
        """Clean up expired memory cache entries"""
        if self.cache_stats['backend'] == 'memory':
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry['expires_at'] <= current_time
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                
    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache system"""
        health = {
            'status': 'healthy',
            'backend': self.cache_stats['backend'],
            'initialized': self._initialized
        }
        
        try:
            if self.redis_client:
                # Test Redis connection
                self.redis_client.ping()
                health['redis_status'] = 'connected'
            else:
                health['redis_status'] = 'not_available'
                health['memory_cache_size'] = len(self.memory_cache)
                
            return health
            
        except Exception as e:
            health['status'] = 'degraded'
            health['error'] = str(e)
            return health

# Global cache manager instance
cache_manager = CacheManager()