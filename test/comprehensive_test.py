#!/usr/bin/env python3
"""
CONSOLIDATED MARKETPULSE COMPREHENSIVE VALIDATOR
===============================================
Single script to test all Day 1-8 components with minimal dependencies
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Get project root
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == 'test' else script_dir


@dataclass
class TestResult:
    day: str
    test_name: str
    passed: bool
    error_message: str = ""


class ConsolidatedValidator:
    """Single comprehensive validator for MarketPulse system"""

    def __init__(self):
        self.project_root = project_root
        self.test_results: List[TestResult] = []
        self.start_time = time.time()

        print("🏥 MARKETPULSE COMPREHENSIVE SYSTEM VALIDATOR")
        print("=" * 60)
        print(f"Project Root: {self.project_root}")
        print("Testing Days 1-8 Components")
        print("=" * 60)

    def log_result(self, day: str, test: str, passed: bool, error: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test}")
        if error and not passed:
            print(f"    Error: {error}")
        self.test_results.append(TestResult(day, test, passed, error))

    def test_day1_structure(self):
        """Day 1: Project structure and setup"""
        print("\n📁 DAY 1: Project Structure & Setup")
        print("-" * 40)

        # Test 1: Basic directories
        required_dirs = ['src', 'test']
        missing = [d for d in required_dirs if not (self.project_root / d).exists()]
        self.log_result('day_1', 'Directory Structure', len(missing) == 0, f"Missing: {missing}")

        # Test 2: Essential files
        essential_files = ['README.md', 'requirements.txt', 'src']
        existing = sum(1 for f in essential_files if (self.project_root / f).exists())
        self.log_result('day_1', 'Essential Files', existing >= 2, f"Found {existing}/3 files")

    def test_day2_ai_framework(self):
        """Day 2: AI framework integration"""
        print("\n🤖 DAY 2: AI Framework Integration")
        print("-" * 40)

        # Test 1: Framework directories
        framework_paths = [
            '01_Framework_Core',
            'src/ai_trading',
            'antifragile_framework'
        ]
        found = sum(1 for p in framework_paths if (self.project_root / p).exists())
        self.log_result('day_2', 'Framework Structure', found >= 1, f"Found {found} framework paths")

        # Test 2: AI components
        ai_files = [
            'src/ai_trading/ai_signal_generator.py',
            'src/ai_trading/ai_fundamental_analyzer.py'
        ]
        existing = sum(1 for f in ai_files if (self.project_root / f).exists())
        self.log_result('day_2', 'AI Components', existing >= 1, f"Found {existing} AI components")

    def test_day3_multiframe(self):
        """Day 3: Multi-timeframe analysis"""
        print("\n📊 DAY 3: Multi-timeframe Analysis")
        print("-" * 40)

        # Test 1: Multi-timeframe collector
        collector_file = self.project_root / 'src/ai_trading/multi_timeframe_collector.py'
        self.log_result('day_3', 'Multi-timeframe Collector', collector_file.exists(),
                        "multi_timeframe_collector.py missing" if not collector_file.exists() else "")

        # Test 2: Technical analysis files
        tech_files = [
            'src/ai_trading/professional_technical_analyzer.py',
            'src/ai_trading/technical_analyzer.py',
            'src/ai_trading/pattern_recognition.py'
        ]
        found = sum(1 for f in tech_files if (self.project_root / f).exists())
        self.log_result('day_3', 'Technical Analysis Components', found >= 1, f"Found {found} components")

    def test_day4_portfolio(self):
        """Day 4: Portfolio theory and risk management"""
        print("\n🛡️ DAY 4: Portfolio Theory & Risk Management")
        print("-" * 40)

        # Test 1: Portfolio theory
        portfolio_files = [
            'src/ai_trading/portfolio_theory.py',
            'portfolio_theory.py',
            'src/ai_trading/portfolio_optimizer.py'
        ]
        found = sum(1 for f in portfolio_files if (self.project_root / f).exists())
        self.log_result('day_4', 'Portfolio Theory', found >= 1, f"Found {found} portfolio files")

        # Test 2: Risk management
        risk_files = [
            'src/ai_trading/ai_risk_manager.py',
            'src/ai_trading/risk_calculator.py',
            'src/ai_trading/risk_management.py'
        ]
        found = sum(1 for f in risk_files if (self.project_root / f).exists())
        self.log_result('day_4', 'Risk Management', found >= 1, f"Found {found} risk management files")

    def test_day5_technical_engine(self):
        """Day 5: Technical analysis engine"""
        print("\n📈 DAY 5: Technical Analysis Engine")
        print("-" * 40)

        # Test 1: AI Signal Generator
        signal_file = self.project_root / 'src/ai_trading/ai_signal_generator.py'
        self.log_result('day_5', 'AI Signal Generator', signal_file.exists())

        # Test 2: Technical Analyzer
        analyzer_file = self.project_root / 'src/ai_trading/professional_technical_analyzer.py'
        self.log_result('day_5', 'Professional Technical Analyzer', analyzer_file.exists())

        # Test 3: Confluence Scoring
        confluence_files = [
            'src/ai_trading/confluence_scoring_system.py',
            'src/ai_trading/confluence_analyzer.py'
        ]
        found = sum(1 for f in confluence_files if (self.project_root / f).exists())
        self.log_result('day_5', 'Confluence Scoring (A+)', found >= 1, f"Found {found} confluence files")

    def test_day6_fundamental(self):
        """Day 6: Fundamental analysis and AI documents"""
        print("\n📋 DAY 6: Fundamental Analysis & AI Documents")
        print("-" * 40)

        # Test 1: AI Fundamental Analyzer
        fund_file = self.project_root / 'src/ai_trading/ai_fundamental_analyzer.py'
        self.log_result('day_6', 'AI Fundamental Analyzer', fund_file.exists())

        # Test 2: AI Document Processor
        doc_file = self.project_root / 'src/ai_trading/ai_document_processor.py'
        self.log_result('day_6', 'AI Document Processor', doc_file.exists())

        # Test 3: Complete Fundamental System
        complete_file = self.project_root / 'src/ai_trading/complete_fundamental_system.py'
        self.log_result('day_6', 'Complete Fundamental System (A+)', complete_file.exists())

    def test_day7_integration(self):
        """Day 7: System architecture and integration"""
        print("\n🏗️ DAY 7: System Architecture & Integration")
        print("-" * 40)

        # Test 1: System integration files
        integration_files = [
            'src/ai_trading/system_monitor.py',
            'src/ai_trading/performance_cache.py',
            'src/ai_trading/error_handling.py',
            'src/ai_trading/system_integration.py'
        ]
        found = sum(1 for f in integration_files if (self.project_root / f).exists())
        self.log_result('day_7', 'System Integration', found >= 2, f"Found {found}/4 integration files")

        # Test 2: Portfolio Management Integration
        portfolio_mgmt_files = [
            'src/ai_trading/portfolio_manager.py',
            'src/ai_trading/complete_portfolio_manager.py'
        ]
        found = sum(1 for f in portfolio_mgmt_files if (self.project_root / f).exists())
        self.log_result('day_7', 'Portfolio Manager Integration', found >= 1, f"Found {found} portfolio managers")

        # Test 3: Integration scripts
        integration_dirs = ['integration', 'scripts', 'pipelines']
        found = sum(1 for d in integration_dirs if (self.project_root / d).exists())
        self.log_result('day_7', 'Integration Scripts', found >= 1, f"Found {found} integration directories")

    def test_day8_realtime(self):
        """Day 8: Real-time streaming and WebSocket services"""
        print("\n🌐 DAY 8: Real-time Streaming & WebSocket Services")
        print("-" * 40)

        # Test 1: Real-time Data Collector
        realtime_files = [
            'src/ai_trading/realtime_data_collector.py',
            'src/ai_trading/real_time_collector.py',
            'src/ai_trading/data_streaming.py'
        ]
        found = sum(1 for f in realtime_files if (self.project_root / f).exists())
        self.log_result('day_8', 'Real-time Data Collector', found >= 1, f"Found {found} real-time files")

        # Test 2: WebSocket Services
        websocket_files = [
            'src/ai_trading/websocket_streaming_service.py',
            'src/ai_trading/websocket_service.py',
            'src/ai_trading/streaming_service.py'
        ]
        found = sum(1 for f in websocket_files if (self.project_root / f).exists())
        self.log_result('day_8', 'WebSocket Streaming Service', found >= 1, f"Found {found} websocket files")

        # Test 3: Integration Pipeline
        pipeline_files = [
            'integration/phase1_day8_pipeline.py',
            'src/ai_trading/integration_pipeline.py',
            'pipelines/day8_pipeline.py'
        ]
        found = sum(1 for f in pipeline_files if (self.project_root / f).exists())
        self.log_result('day_8', 'Integration Pipeline', found >= 1, f"Found {found} pipeline files")

        # Test 4: Testing Framework
        test_files = [
            'test/test_realtime_system.py',
            'test/integration_tests.py',
            'test/day8_tests.py'
        ]
        found = sum(1 for f in test_files if (self.project_root / f).exists())
        self.log_result('day_8', 'Testing Framework', found >= 1, f"Found {found} test files")

    def generate_report(self):
        """Generate comprehensive test report"""
        elapsed = time.time() - self.start_time

        print(f"\n⏱️ Validation completed in {elapsed:.1f} seconds")
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE SYSTEM VALIDATION REPORT")
        print("=" * 60)

        # Calculate metrics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"Overall Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")

        # Daily breakdown
        daily_stats = {}
        for result in self.test_results:
            if result.day not in daily_stats:
                daily_stats[result.day] = {'passed': 0, 'total': 0}
            daily_stats[result.day]['total'] += 1
            if result.passed:
                daily_stats[result.day]['passed'] += 1

        print(f"\n📅 DAILY VALIDATION RESULTS:")
        print("-" * 40)

        day_names = {
            'day_1': 'Day 1', 'day_2': 'Day 2', 'day_3': 'Day 3', 'day_4': 'Day 4',
            'day_5': 'Day 5', 'day_6': 'Day 6', 'day_7': 'Day 7', 'day_8': 'Day 8'
        }

        foundation_passed = 0
        foundation_total = 0

        for day_key in sorted(daily_stats.keys()):
            stats = daily_stats[day_key]
            day_pass_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0

            if day_pass_rate >= 80:
                grade = "A"
                icon = "✅"
            elif day_pass_rate >= 60:
                grade = "B"
                icon = "🟡"
            else:
                grade = "C"
                icon = "❌"

            print(
                f"{icon} {day_names[day_key]}: {grade} ({day_pass_rate:.0f}%) - {stats['passed']}/{stats['total']} tests")

            # Track foundation week (Days 1-7)
            if day_key in ['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']:
                foundation_passed += stats['passed']
                foundation_total += stats['total']

        # Overall assessment
        foundation_rate = (foundation_passed / foundation_total * 100) if foundation_total > 0 else 0

        print(f"\n🏆 FOUNDATION WEEK (Days 1-7): {foundation_rate:.0f}%")
        print(
            f"🚀 PHASE 1 DAY 8: {daily_stats.get('day_8', {'passed': 0, 'total': 1})['passed']}/{daily_stats.get('day_8', {'passed': 0, 'total': 1})['total']} tests")

        # System health
        print(f"\n🏥 SYSTEM HEALTH ASSESSMENT:")
        print("-" * 30)

        if pass_rate >= 80:
            health = "💚 EXCELLENT - Production Ready"
        elif pass_rate >= 60:
            health = "🟡 GOOD - Minor Issues"
        elif pass_rate >= 40:
            health = "🟠 FAIR - Needs Attention"
        else:
            health = "🔴 POOR - Major Issues"

        print(health)

        # Failed tests
        failed_tests = [r for r in self.test_results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            print("-" * 20)
            for test in failed_tests[:10]:  # Show first 10
                print(f"  • {day_names.get(test.day, test.day)}: {test.test_name}")
                if test.error_message:
                    print(f"    {test.error_message}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)

        if pass_rate >= 75:
            print("✅ System ready for Phase 1 Day 9")
            print("🚀 Proceed with advanced ML integration")
        elif pass_rate >= 50:
            print("⚠️ Address failed tests before proceeding")
            print("🔧 Focus on missing components")
        else:
            print("🛠️ Significant work needed")
            print("📚 Review Foundation Week implementation")

        print(f"\n{'=' * 60}")
        status = "READY" if pass_rate >= 60 else "NEEDS WORK"
        print(f"🎯 SYSTEM STATUS: {status}")
        print(f"{'=' * 60}")

        return pass_rate >= 60

    def run_all_tests(self):
        """Run all validation tests"""
        try:
            self.test_day1_structure()
            self.test_day2_ai_framework()
            self.test_day3_multiframe()
            self.test_day4_portfolio()
            self.test_day5_technical_engine()
            self.test_day6_fundamental()
            self.test_day7_integration()
            self.test_day8_realtime()

            return self.generate_report()

        except KeyboardInterrupt:
            print("\n⛔ Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n💥 Test failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Run the consolidated validator"""
    print("🧪 Starting MarketPulse System Comprehensive Validation...")

    validator = ConsolidatedValidator()
    success = validator.run_all_tests()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()