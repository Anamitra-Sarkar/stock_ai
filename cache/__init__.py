"""Cache management package"""
from .redis_cache import CacheManager

cache_manager = CacheManager()

__all__ = ['cache_manager', 'CacheManager']
