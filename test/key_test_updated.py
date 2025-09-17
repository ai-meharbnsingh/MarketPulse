# MarketPulse Key Test - Simple Version
# File: 02_test/key_test_simple.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


def main():
    print("🔍 MarketPulse API Key Check")
    print("=" * 40)

    # Check .env file
    env_file = project_root / ".env"
    print(f"📁 Project root: {project_root}")
    print(f"📄 .env file: {'✅ Found' if env_file.exists() else '❌ Missing'}")
    print()

    # Check each provider
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Gemini": "GEMINI_API_KEY"
    }

    total_keys = 0
    all_good = True

    for name, var in providers.items():
        raw_value = os.getenv(var, "")

        if not raw_value:
            print(f"❌ {name}: No {var} found")
            all_good = False
            continue

        keys = [k.strip() for k in raw_value.split(",") if k.strip()]
        total_keys += len(keys)

        print(f"✅ {name}: {len(keys)} keys loaded")

        # Show first few characters for verification
        for i, key in enumerate(keys, 1):
            preview = f"{key[:8]}..." if len(key) > 8 else "***"
            print(f"   Key {i}: {preview}")

    print()
    print(f"📊 Total keys: {total_keys}")

    # Check framework settings
    print()
    print("⚙️ Framework settings:")
    settings = ["FAST_DEMO_MODE", "PERFORMANCE_TEST_MODE", "LOG_LEVEL"]
    for setting in settings:
        value = os.getenv(setting, "not set")
        print(f"   {setting}: {value}")

    print()
    if all_good and total_keys > 0:
        print("🎉 All API keys loaded successfully!")
        print("✅ Ready for framework testing")
        print()
        print("Next step: python 02_test/framework_test_simple.py")
    else:
        print("❌ Some API keys missing")
        print("Fix your .env file first")

    return all_good


if __name__ == "__main__":
    main()