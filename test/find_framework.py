# Framework Finder Script
# File: 02_test/find_framework.py

import os
from pathlib import Path


def find_framework():
    print("🔍 MarketPulse Framework Finder")
    print("=" * 40)

    project_root = Path(__file__).parent.parent
    print(f"📁 Project root: {project_root}")
    print()

    # Check current directory contents
    print("📂 Project structure:")
    for item in sorted(project_root.iterdir()):
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")

    print()

    # Look for antifragile_framework
    potential_locations = [
        project_root / "antifragile_framework",
        project_root / "01_Framework" / "antifragile_framework",
        project_root / "src" / "antifragile_framework",
        project_root / "framework" / "antifragile_framework"
    ]

    print("🔍 Searching for antifragile_framework...")
    framework_found = False

    for location in potential_locations:
        if location.exists():
            print(f"✅ Found at: {location}")

            # Check if it has the required modules
            required_modules = [
                "core/failover_engine.py",
                "config/config_loader.py",
                "providers/api_abstraction_layer.py",
                "providers/provider_registry.py"
            ]

            print("   📋 Checking required modules:")
            all_modules_found = True
            for module in required_modules:
                module_path = location / module
                if module_path.exists():
                    print(f"      ✅ {module}")
                else:
                    print(f"      ❌ {module}")
                    all_modules_found = False

            if all_modules_found:
                print(f"   🎉 Complete framework found at: {location}")
                framework_found = True
                return str(location.parent)
            else:
                print(f"   ⚠️  Incomplete framework at: {location}")
        else:
            print(f"❌ Not found at: {location}")

    # Search recursively
    if not framework_found:
        print("\n🔍 Searching recursively...")
        for root, dirs, files in os.walk(project_root):
            if "antifragile_framework" in dirs:
                found_path = Path(root) / "antifragile_framework"
                print(f"✅ Found framework folder at: {found_path}")
                return str(Path(root))

    # Look for telemetry too
    print("\n🔍 Searching for telemetry...")
    telemetry_locations = [
        project_root / "telemetry",
        project_root / "01_Framework" / "telemetry"
    ]

    for location in telemetry_locations:
        if location.exists():
            print(f"✅ Telemetry found at: {location}")
        else:
            print(f"❌ Telemetry not found at: {location}")

    if not framework_found:
        print("\n❌ antifragile_framework not found!")
        print("\n💡 Possible solutions:")
        print("1. Check if you have the framework files in your project")
        print("2. Make sure the folder structure matches your uploads")
        print("3. The framework might be in a different location")

    return None


if __name__ == "__main__":
    framework_path = find_framework()

    if framework_path:
        print(f"\n🎯 Framework base path: {framework_path}")
        print("\n📝 Add this to your test script:")
        print(f'sys.path.insert(0, r"{framework_path}")')
    else:
        print("\n❌ Framework not found. Check your project structure.")