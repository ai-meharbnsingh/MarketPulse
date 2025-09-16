
# AI Provider Performance Optimizer
import time
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ProviderMetrics:
    """Track performance metrics for AI providers"""
    response_time: float = 0.0
    success_rate: float = 1.0
    cost_per_token: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_used: float = 0.0

class AIProviderOptimizer:
    """Optimize AI provider selection based on performance and cost"""

    def __init__(self):
        self.provider_metrics: Dict[str, ProviderMetrics] = {
            'openai': ProviderMetrics(cost_per_token=0.002),
            'anthropic': ProviderMetrics(cost_per_token=0.008),
            'gemini': ProviderMetrics(cost_per_token=0.001)
        }
        self.daily_budget = 50.0  # $50 daily budget
        self.daily_spent = 0.0

    def update_provider_metrics(self, provider: str, response_time: float, 
                              success: bool, tokens_used: int):
        """Update performance metrics for a provider"""
        if provider not in self.provider_metrics:
            return

        metrics = self.provider_metrics[provider]
        metrics.total_requests += 1
        metrics.last_used = time.time()

        if success:
            # Update average response time
            metrics.response_time = (
                (metrics.response_time * (metrics.total_requests - 1) + response_time) 
                / metrics.total_requests
            )
        else:
            metrics.failed_requests += 1

        # Update success rate
        metrics.success_rate = (
            (metrics.total_requests - metrics.failed_requests) / metrics.total_requests
        )

        # Update daily spending
        cost = tokens_used * metrics.cost_per_token
        self.daily_spent += cost

    def get_best_provider(self) -> Optional[str]:
        """Select the best provider based on performance and budget"""
        if self.daily_spent >= self.daily_budget:
            return None  # Budget exceeded

        # Score providers based on success rate, response time, and cost
        best_provider = None
        best_score = -1

        for provider, metrics in self.provider_metrics.items():
            if metrics.total_requests == 0:
                # Give new providers a chance
                score = 0.5
            else:
                # Weighted score: success_rate * 0.5 + speed_score * 0.3 + cost_score * 0.2
                speed_score = max(0, 1 - (metrics.response_time / 10))  # Normalize to 0-1
                cost_score = max(0, 1 - (metrics.cost_per_token / 0.01))  # Normalize to 0-1

                score = (metrics.success_rate * 0.5 + 
                        speed_score * 0.3 + 
                        cost_score * 0.2)

            if score > best_score:
                best_score = score
                best_provider = provider

        return best_provider

    def get_provider_status(self) -> Dict:
        """Get current status of all providers"""
        return {
            'providers': dict(self.provider_metrics),
            'daily_budget': self.daily_budget,
            'daily_spent': self.daily_spent,
            'budget_remaining': self.daily_budget - self.daily_spent
        }

# Global optimizer instance
ai_optimizer = AIProviderOptimizer()
