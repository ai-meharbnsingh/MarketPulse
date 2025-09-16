
# Memory Optimization Utilities for MarketPulse
import gc
import sys
import weakref
from typing import Dict, Any, List
from functools import wraps
import pandas as pd

class MemoryManager:
    """Manage memory usage and cleanup for MarketPulse"""

    def __init__(self):
        self.tracked_objects = weakref.WeakSet()
        self.large_dataframes = {}

    def track_object(self, obj: Any, name: str = None):
        """Track an object for memory management"""
        self.tracked_objects.add(obj)
        if isinstance(obj, pd.DataFrame) and len(obj) > 1000:
            obj_name = name or f"dataframe_{id(obj)}"
            self.large_dataframes[obj_name] = {
                'shape': obj.shape,
                'memory_mb': obj.memory_usage(deep=True).sum() / 1024 / 1024
            }

    def cleanup_dataframes(self):
        """Clean up large DataFrames to free memory"""
        for name, info in list(self.large_dataframes.items()):
            print(f"   Cleaned up {name}: {info['shape']} ({info['memory_mb']:.1f} MB)")

        self.large_dataframes.clear()
        gc.collect()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        # Get process memory info
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'virtual_memory_mb': memory_info.vms / 1024 / 1024,
                'tracked_objects': len(self.tracked_objects),
                'large_dataframes': len(self.large_dataframes),
                'total_df_memory_mb': sum(df['memory_mb'] for df in self.large_dataframes.values())
            }
        except ImportError:
            return {
                'tracked_objects': len(self.tracked_objects),
                'large_dataframes': len(self.large_dataframes)
            }

    def force_cleanup(self):
        """Force memory cleanup and garbage collection"""
        # Clear tracked objects
        self.tracked_objects.clear()
        self.cleanup_dataframes()

        # Force garbage collection
        collected = gc.collect()

        return {
            'objects_collected': collected,
            'memory_after_cleanup': self.get_memory_info()
        }

def memory_efficient(cleanup_after: bool = True):
    """Decorator for memory-efficient function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record memory before
            memory_before = memory_manager.get_memory_info()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    gc.collect()

                memory_after = memory_manager.get_memory_info()

                # Log memory usage if significant
                if 'process_memory_mb' in memory_before and 'process_memory_mb' in memory_after:
                    memory_diff = memory_after['process_memory_mb'] - memory_before['process_memory_mb']
                    if abs(memory_diff) > 50:  # More than 50MB difference
                        print(f"   Memory change in {func.__name__}: {memory_diff:+.1f} MB")

        return wrapper
    return decorator

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage"""
    if df is None or df.empty:
        return df

    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Optimize object columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')

    return df

def batch_process_data(data_list: List[Any], batch_size: int = 100, 
                      process_func: callable = None):
    """Process data in batches to manage memory usage"""
    if not process_func:
        return data_list

    results = []

    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch_result = process_func(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

        # Cleanup after each batch
        if i % (batch_size * 5) == 0:  # Every 5 batches
            gc.collect()

    return results

# Global memory manager
memory_manager = MemoryManager()

# Context manager for memory-conscious operations
class MemoryContext:
    """Context manager for memory-conscious operations"""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.initial_memory = None

    def __enter__(self):
        self.initial_memory = memory_manager.get_memory_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = memory_manager.get_memory_info()
        gc.collect()

        if ('process_memory_mb' in self.initial_memory and 
            'process_memory_mb' in final_memory):
            memory_diff = (final_memory['process_memory_mb'] - 
                          self.initial_memory['process_memory_mb'])
            if abs(memory_diff) > 10:  # More than 10MB difference
                print(f"   Memory usage for {self.operation_name}: {memory_diff:+.1f} MB")
