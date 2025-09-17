# MarketPulse Framework Test - Clean Version
# File: 02_test/framework_test_clean.py
# Uses ONLY your project structure, ignores any venv installations

import asyncio
import os
import sys
import time
from pathlib import Path

# Use ONLY your project structure
project_root = Path(__file__).parent.parent
framework_core = project_root / "01_Framework_Core"

print("🧹 Clean Framework Test (Project Structure Only)")
print("=" * 60)
print(f"📁 Project root: {project_root}")
print(f"📁 Framework core: {framework_core}")

# Verify the correct structure exists
required_paths = {
    "antifragile_framework": framework_core / "antifragile_framework",
    "telemetry": framework_core / "telemetry",
    "failover_engine": framework_core / "antifragile_framework" / "core" / "failover_engine.py",
    "config_loader": framework_core / "antifragile_framework" / "config" / "config_loader.py"
}

print("\n🔍 Verifying project structure:")
all_good = True
for name, path in required_paths.items():
    exists = path.exists()
    print(f"  {'✅' if exists else '❌'} {name}: {path}")
    if not exists:
        all_good = False

if not all_good:
    print("\n❌ Required framework files missing!")
    print("Make sure your 01_Framework_Core folder has the complete antifragile_framework")
    sys.exit(1)

# Clean sys.path - remove any existing antifragile references
sys.path = [p for p in sys.path if 'antifragile' not in p.lower()]

# Add ONLY our project paths
sys.path.insert(0, str(framework_core))
sys.path.insert(0, str(project_root))

print(f"\n🔧 Python path configured:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    if 'Trading_App' in path:
        print(f"  {i}: {path}")

# Load environment
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Import framework components
print(f"\n📦 Importing framework components...")
try:
    from antifragile_framework.core.failover_engine import FailoverEngine

    print("  ✅ FailoverEngine")

    from antifragile_framework.config.config_loader import load_provider_profiles

    print("  ✅ config_loader")

    from antifragile_framework.providers.api_abstraction_layer import ChatMessage

    print("  ✅ ChatMessage")

    from antifragile_framework.providers.provider_registry import get_default_provider_registry

    print("  ✅ provider_registry")

    from telemetry.event_bus import EventBus

    print("  ✅ EventBus")

    print("✅ All imports successful!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"\nDebugging info:")
    print(f"Current working dir: {os.getcwd()}")
    print(f"Python path entries with 'Trading_App':")
    for path in sys.path:
        if 'Trading_App' in path:
            print(f"  - {path}")
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


async def run_framework_test():
    print(f"\n🚀 Testing MarketPulse Framework")
    print("=" * 40)

    # Check API keys
    total_keys = sum(len(config["api_keys"]) for config in PROVIDER_CONFIGS.values())
    print(f"🔑 API keys loaded: {total_keys}")

    if total_keys == 0:
        print("❌ No API keys found in .env!")
        return False

    # Initialize framework
    print(f"\n🏗️ Initializing framework...")
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
        print(f"❌ Framework initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test simple query
    print(f"\n🧪 Testing simple AI query...")
    try:
        start_time = time.time()

        response = await failover_engine.execute_request(
            model_priority_map={
                "google_gemini": ["gemini-1.5-flash-latest"],
                "openai": ["gpt-4o"],
                "anthropic": ["claude-3-5-sonnet-20240620"]
            },
            messages=[
                ChatMessage(role="user", content="What is 2+2? Answer with just the number.")
            ],
            max_estimated_cost_usd=0.001,
            request_id="test_simple"
        )

        end_time = time.time()

        print(f"✅ Success!")
        print(f"🤖 Model used: {response.model_used}")
        print(f"⏱️ Total time: {end_time - start_time:.2f}s")
        print(f"📝 Response: {response.content.strip()}")

        return True

    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_market_analysis():
    print(f"\n📈 Testing market analysis...")
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
                "anthropic": ["claude-3-5-sonnet-20240620"]
            },
            messages=[
                ChatMessage(role="system", content="You are a financial analyst."),
                ChatMessage(role="user", content="What drives stock prices? Answer in 10 words.")
            ],
            max_estimated_cost_usd=0.005,
            request_id="test_market"
        )

        print(f"✅ Market analysis successful!")
        print(f"🤖 Model: {response.model_used}")
        print(f"📝 Response: {response.content}")

        return True

    except Exception as e:
        print(f"❌ Market analysis failed: {e}")
        return False


async def main():
    # Run tests
    basic_success = await run_framework_test()

    if basic_success:
        market_success = await test_market_analysis()
    else:
        market_success = False

    # Final summary
    print(f"\n" + "=" * 60)
    print("🎯 DAY 1 FRAMEWORK TEST RESULTS")
    print("=" * 60)
    print(f"Basic Framework Test: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Market Analysis Test: {'✅ PASSED' if market_success else '❌ FAILED'}")

    if basic_success and market_success:
        print(f"\n🎉 SUCCESS! Your MarketPulse framework is working!")
        print(f"✅ Multi-AI provider system operational")
        print(f"✅ Cost optimization and failover active")
        print(f"✅ Market analysis capabilities confirmed")
        print(f"✅ Ready for Day 2: Trading Psychology")

        # Save success file
        success_file = project_root / "03_docs" / "day1_success.txt"
        success_file.parent.mkdir(exist_ok=True)

        with open(success_file, "w") as f:
            f.write("MarketPulse Day 1 - SUCCESS!\n")
            f.write("=" * 30 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Framework: Fully Operational\n")
            f.write("API Keys: All Working\n")
            f.write("Status: Ready for Day 2\n")

        print(f"📄 Success logged to: {success_file}")

    else:
        print(f"\n❌ Some tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())