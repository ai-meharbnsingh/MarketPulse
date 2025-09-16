# scripts/fix_encoding_issues.py
"""
Fix Character Encoding Issues for Windows
=========================================

Fixes the Unicode/emoji encoding issues that are causing problems
on Windows systems with charmap codec errors.
"""

import os
import sys
from pathlib import Path
import re


def fix_file_encoding(file_path, encoding_from='utf-8', encoding_to='utf-8'):
    """Fix file encoding by removing problematic Unicode characters"""
    try:
        # Read with UTF-8
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Replace problematic Unicode characters with ASCII equivalents
        emoji_replacements = {
            '🚀': '[ROCKET]',
            '✅': '[CHECK]',
            '❌': '[X]',
            '⚠️': '[WARNING]',
            '📊': '[CHART]',
            '🎯': '[TARGET]',
            '🏆': '[TROPHY]',
            '🎉': '[PARTY]',
            '📋': '[CLIPBOARD]',
            '🔧': '[WRENCH]',
            '🧠': '[BRAIN]',
            '🛡️': '[SHIELD]',
            '🧹': '[BROOM]',
            '📦': '[PACKAGE]',
            '🏗️': '[CONSTRUCTION]',
            '🧪': '[TEST_TUBE]',
            '⚡': '[LIGHTNING]',
            '🏁': '[CHECKERED_FLAG]',
            '📝': '[MEMO]',
            '📈': '[CHART_UP]',
            '🔮': '[CRYSTAL_BALL]',
            '💡': '[BULB]',
            '🎊': '[CONFETTI]',
            '🎓': '[GRADUATION]',
            '🏅': '[MEDAL]',
            '📁': '[FOLDER]',
            '🔄': '[REFRESH]',
            '✨': '[SPARKLES]'
        }

        # Replace emojis
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)

        # Write back with proper encoding
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

        print(f"   [CHECK] Fixed encoding for {file_path}")
        return True

    except Exception as e:
        print(f"   [X] Could not fix {file_path}: {e}")
        return False


def main():
    """Fix encoding issues in all Python files"""
    print("[ROCKET] Fixing Character Encoding Issues...")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # Files that need encoding fixes
    files_to_fix = [
        'scripts/day7_performance_optimizer.py',
        'scripts/day7_complete_system_test.py',
        'scripts/day7_session_complete.py',
        'src/ai_trading/performance_cache.py',
        'src/ai_trading/ai_optimizer.py',
        'src/ai_trading/error_handling.py',
        'src/ai_trading/system_monitor.py',
        'src/ai_trading/memory_optimizer.py'
    ]

    fixed_count = 0

    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_file_encoding(full_path):
                fixed_count += 1
        else:
            print(f"   [WARNING] File not found: {file_path}")

    print(f"\n[TARGET] Encoding Fix Summary:")
    print(f"Files fixed: {fixed_count}/{len(files_to_fix)}")

    if fixed_count == len(files_to_fix):
        print("[CHECK] All encoding issues resolved!")
    else:
        print("[WARNING] Some files still have issues")

    return fixed_count > 0


if __name__ == "__main__":
    main()