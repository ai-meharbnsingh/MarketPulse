
# Performance Caching System for MarketPulse
import functools
import time
from typing import Any, Dict

class MarketPulseCache:
    """Simple in-memory cache for expensive operations"""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any:
        """Get value from cache if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                return entry['value']
            else:
                del self.cache[key]  # Remove expired entry
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp"""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

    def clear(self) -> None:
        """Clear all cached values"""
        self.cache.clear()

# Global cache instance
_global_cache = MarketPulseCache()

def cached_analysis(cache_key_prefix: str):
    """Decorator for caching expensive analysis operations"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{cache_key_prefix}_{func.__name__}_{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result)
            return result
        return wrapper
    return decorator

# Usage example:
# @cached_analysis("fundamental")
# def analyze_stock_fundamentals(symbol: str):
#     # Expensive fundamental analysis
#     return results
