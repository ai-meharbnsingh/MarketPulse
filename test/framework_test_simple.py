# MarketPulse Framework Test - Simple Version
# File: 02_test/framework_test_simple.py

import asyncio
import os
import sys
import time
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Try to import framework
try:
    from antifragile_framework.core.failover_engine import FailoverEngine
    from antifragile_framework.config.config_loader import load_provider_profiles
    from antifragile_framework.providers.api_abstraction_layer import ChatMessage
    from antifragile_framework.providers.provider_registry import get_default_provider_registry
    from telemetry.event_bus import EventBus

    print("✅ Framework imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative path...")

    # Try 01_Framework path
    framework_path = project_root / "01_Framework"
    if framework_path.exists():
        sys.path.insert(0, str(framework_path))
        try:
            from antifragile_framework.core.failover_engine import FailoverEngine
            from antifragile_framework.config.config_loader import load_provider_profiles
            from antifragile_framework.providers.api_abstraction_layer import ChatMessage
            from antifragile_framework.providers.provider_registry import get_default_provider_registry
            from telemetry.event_bus import EventBus

            print("✅ Framework imports successful from 01_Framework!")
        except ImportError as e2:
            print(f"❌ Still failed: {e2}")
            sys.exit(1)
    else:
        print("❌ Framework not found")
        sys.exit(1)

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


async def test_framework():
    print("🚀 MarketPulse Framework Test")
    print("=" * 40)

    # Check API keys
    total_keys = sum(len(config["api_keys"]) for config in PROVIDER_CONFIGS.values())
    print(f"📊 API keys loaded: {total_keys}")

    if total_keys == 0:
        print("❌ No API keys found!")
        return False

    # Initialize framework
    print("\n🏗️ Initializing framework...")
    try:
        provider_profiles = load_provider_profiles()
        provider_registry = get_default_provider_registry()
        event_bus = EventBus()

        failover_engine = FailoverEngine(
            provider_configs=PROVIDER_CONFIGS,
            provider_registry=provider_registry,
            event_bus=event_bus,
            provider_profiles=provider_profiles,
        )
        print("✅ Framework initialized!")
    except Exception as e:
        print(f"❌ Framework init failed: {e}")
        return False

    # Test simple query
    print("\n🧪 Testing simple query...")
    try:
        start_time = time.time()

        response = await failover_engine.execute_request(
            model_priority_map={
                "google_gemini": ["gemini-1.5-flash-latest"],
                "openai": ["gpt-4o"],
                "anthropic": ["claude-3-5-sonnet-20240620"]
            },
            messages=[
                ChatMessage(role="user", content="What is 2+2? One word answer.")
            ],
            max_estimated_cost_usd=0.001,
            request_id="simple_test"
        )

        end_time = time.time()

        print(f"✅ Query successful!")
        print(f"🤖 Model: {response.model_used}")
        print(f"⏱️ Time: {end_time - start_time:.2f}s")
        print(f"📝 Answer: {response.content}")

        return True

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False


async def test_market_query():
    print("\n📈 Testing market query...")
    try:
        provider_profiles = load_provider_profiles()
        provider_registry = get_default_provider_registry()
        event_bus = EventBus()

        failover_engine = FailoverEngine(
            provider_configs=PROVIDER_CONFIGS,
            provider_registry=provider_registry,
            event_bus=event_bus,
            provider_profiles=provider_profiles,
        )

        response = await failover_engine.execute_request(
            model_priority_map={
                "google_gemini": ["gemini-1.5-flash-latest"],
                "anthropic": ["claude-3-5-sonnet-20240620"],
                "openai": ["gpt-4o"]
            },
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a financial analyst."
                ),
                ChatMessage(
                    role="user",
                    content="What causes stock volatility? Answer in 15 words."
                )
            ],
            max_estimated_cost_usd=0.005,
            request_id="market_test"
        )

        print(f"✅ Market query successful!")
        print(f"🤖 Model: {response.model_used}")
        print(f"📝 Answer: {response.content}")

        return True

    except Exception as e:
        print(f"❌ Market query failed: {e}")
        return False


async def main():
    print("🚀 Starting MarketPulse Framework Tests")
    print(f"📁 Location: 02_test/framework_test_simple.py")
    print(f"📁 Project: {project_root}")
    print()

    # Run tests
    basic_test = await test_framework()

    if basic_test:
        market_test = await test_market_query()
    else:
        market_test = False

    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    print(f"Basic Framework: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    print(f"Market Analysis: {'✅ PASSED' if market_test else '❌ FAILED'}")

    if basic_test and market_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ MarketPulse framework is operational!")
        print("✅ Ready for Day 2: Trading Psychology")

        # Save simple results
        docs_dir = project_root / "03_docs"
        docs_dir.mkdir(exist_ok=True)

        with open(docs_dir / "day1_results.txt", "w") as f:
            f.write("Day 1 Framework Test Results\n")
            f.write("============================\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Basic Framework: PASSED\n")
            f.write("Market Analysis: PASSED\n")
            f.write("\nStatus: Ready for Day 2\n")

        print(f"📄 Results saved to: 03_docs/day1_results.txt")

    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())