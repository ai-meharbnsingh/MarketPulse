# File: 02_test/quick_test_day3.py

import asyncio
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "01_Framework_Core"))

from antifragile_framework.core.failover_engine import FailoverEngine
from antifragile_framework.config.config_loader import load_provider_profiles
from antifragile_framework.providers.api_abstraction_layer import ChatMessage
from antifragile_framework.providers.provider_registry import get_default_provider_registry
from telemetry.event_bus import EventBus
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROVIDER_CONFIGS = {
    "openai": {
        "api_keys": [k.strip() for k in os.getenv("OPENAI_API_KEY", "").split(",") if k.strip()],
        "resource_config": {},
        "circuit_breaker_config": {},
    },
    "google_gemini": {
        "api_keys": [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()],
        "resource_config": {},
        "circuit_breaker_config": {},
    },
    "anthropic": {
        "api_keys": [k.strip() for k in os.getenv("ANTHROPIC_API_KEY", "").split(",") if k.strip()],
        "resource_config": {},
        "circuit_breaker_config": {},
    },
}


async def test_framework_ready():
    print("🎯 Day 3 Framework Readiness Test")
    print("=" * 40)

    # Initialize framework
    provider_profiles = load_provider_profiles()
    provider_registry = get_default_provider_registry()
    event_bus = EventBus()

    failover_engine = FailoverEngine(
        provider_configs=PROVIDER_CONFIGS,
        provider_registry=provider_registry,
        event_bus=event_bus,
        provider_profiles=provider_profiles,
    )

    # Test with reasonable cost cap
    response = await failover_engine.execute_request(
        model_priority_map={
            "google_gemini": ["gemini-1.5-flash-latest"],
            "openai": ["gpt-4o"],
            "anthropic": ["claude-3-5-sonnet-20240620"]
        },
        messages=[
            ChatMessage(role="user", content="What is technical analysis? Reply in 5 words.")
        ],
        max_estimated_cost_usd=0.01,  # Higher cost cap
        request_id="day3_ready_test"
    )

    print(f"✅ Framework operational!")
    print(f"🤖 Model: {response.model_used}")
    print(f"📝 Response: {response.content}")
    print(f"🎉 Ready for Day 3 pattern recognition!")

    return True


if __name__ == "__main__":
    asyncio.run(test_framework_ready())