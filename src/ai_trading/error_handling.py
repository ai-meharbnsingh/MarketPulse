
# Enhanced Error Handling for MarketPulse
import logging
import functools
import traceback
from typing import Any, Optional, Callable
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class MarketPulseError(Exception):
    """Base exception for MarketPulse system"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.severity = severity
        super().__init__(message)

class DataError(MarketPulseError):
    """Data-related errors (missing data, API failures, etc.)"""
    pass

class AIError(MarketPulseError):
    """AI-related errors (provider failures, API limits, etc.)"""
    pass

class AnalysisError(MarketPulseError):
    """Analysis-related errors (calculation failures, invalid inputs, etc.)"""
    pass

def safe_execute(fallback_value: Any = None, 
                error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                log_errors: bool = True):
    """Decorator for safe execution with fallback values"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {str(e)}")
                    if error_severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                        logging.error(f"Traceback: {traceback.format_exc()}")

                # Return fallback value
                return fallback_value
        return wrapper
    return decorator

def graceful_degradation(fallback_func: Optional[Callable] = None):
    """Decorator for graceful degradation when primary function fails"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Primary function {func.__name__} failed: {str(e)}")

                if fallback_func:
                    try:
                        logging.info(f"Attempting fallback function for {func.__name__}")
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logging.error(f"Fallback also failed: {str(fallback_error)}")
                        raise MarketPulseError(
                            f"Both primary and fallback functions failed for {func.__name__}",
                            ErrorSeverity.HIGH
                        )
                else:
                    raise MarketPulseError(
                        f"Function {func.__name__} failed with no fallback",
                        ErrorSeverity.MEDIUM
                    )
        return wrapper
    return decorator

class ErrorRecoveryManager:
    """Manage error recovery strategies"""

    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}

    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, context: str = "") -> Any:
        """Handle error with appropriate recovery strategy"""
        error_type = type(error)
        error_key = f"{error_type.__name__}_{context}"

        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Apply recovery strategy if available
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logging.error(f"Recovery strategy failed: {str(recovery_error)}")

        # Default handling
        if self.error_counts[error_key] > 3:
            raise MarketPulseError(
                f"Repeated error in {context}: {str(error)}",
                ErrorSeverity.HIGH
            )

        return None

# Global error recovery manager
error_manager = ErrorRecoveryManager()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
