"""
Cache module for the enterprise stock platform
"""
from .redis_cache import cache_manager

__all__ = ['cache_manager']