# 02_test/phase1_day8_validation.py
"""
Phase 1 Day 8 - Real-Time Data Pipeline Testing & Validation
Comprehensive testing of the streaming architecture integration

Test Categories:
1. Real-time data collection functionality
2. WebSocket streaming service
3. AI-enhanced analysis pipeline
4. Integration with Foundation Week components
5. Performance and reliability testing
"""

import asyncio
import pytest
import logging
import json
import time
import websockets
from datetime import datetime, timezone
from typing import Dict, List, Any
import threading
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.integration.phase1_day8_pipeline import RealTimeMarketIntelligence
from src.data.collectors.realtime_market_data import RealTimeDataCollector, MarketDataPoint
from src.data.streaming.websocket_service import WebSocketStreamingService
from antifragile_framework import FrameworkAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase1Day8Validator:
    """Comprehensive validator for Phase 1 Day 8 implementation"""

    def __init__(self):
        """Initialize the validator"""
        self.test_results = {}
        self.performance_metrics = {}
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

        # Initialize components for testing
        self.ai_framework = None
        self.data_collector = None
        self.websocket_service = None
        self.intelligence_system = None

        logger.info("🧪 Phase 1 Day 8 Validator Initialized")

    async def run_comprehensive_tests(self):
        """Run all validation tests"""
        logger.info("🎯 Starting Phase 1 Day 8 Comprehensive Validation")
        logger.info("=" * 60)

        test_suite = [
            ("AI Framework Setup", self.test_ai_framework_setup),
            ("Real-Time Data Collector", self.test_realtime_data_collector),
            ("WebSocket Streaming Service", self.test_websocket_streaming),
            ("Foundation Integration", self.test_foundation_integration),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling & Recovery", self.test_error_handling),
            ("Real-Time Analysis Quality", self.test_analysis_quality)
        ]

        total_tests = len(test_suite)
        passed_tests = 0

        for test_name, test_function in test_suite:
            logger.info(f"\n🧪 Running Test: {test_name}")
            logger.info("-" * 40)

            try:
                start_time = time.time()
                result = await test_function()
                test_duration = time.time() - start_time

                if result:
                    logger.info(f"✅ {test_name}: PASSED ({test_duration:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED ({test_duration:.2f}s)")

                self.test_results[test_name] = {
                    'passed': result,
                    'duration': test_duration,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = {
                    'passed': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

        # Generate final report
        success_rate = passed_tests / total_tests
        await self.generate_validation_report(success_rate, passed_tests, total_tests)

        return success_rate >= 0.8  # 80% pass rate required

    async def test_ai_framework_setup(self) -> bool:
        """Test AI framework initialization and basic functionality"""
        try:
            logger.info("🤖 Testing AI Framework setup...")

            # Initialize AI framework
            self.ai_framework = FrameworkAPI()

            # Test basic connectivity
            test_prompt = "Respond with 'FRAMEWORK_TEST_OK' to confirm connectivity"
            response = await asyncio.wait_for(
                self.ai_framework.get_completion(test_prompt),
                timeout=15.0
            )

            if "FRAMEWORK_TEST_OK" in response or "ok" in response.lower():
                logger.info("✅ AI Framework connectivity confirmed")
                return True
            else:
                logger.error(f"❌ Unexpected AI response: {response}")
                return False

        except Exception as e:
            logger.error(f"AI Framework test failed: {e}")
            return False

    async def test_realtime_data_collector(self) -> bool:
        """Test real-time data collection functionality"""
        try:
            logger.info("📡 Testing Real-Time Data Collector...")

            # Initialize data collector
            self.data_collector = RealTimeDataCollector(self.ai_framework)

            # Test data collection for a short period
            collected_data = []

            def data_handler(data_point: MarketDataPoint):
                collected_data.append(data_point)
                logger.info(f"📊 Collected: {data_point.symbol} @ ₹{data_point.price:.2f}")

            self.data_collector.add_subscriber(data_handler)

            # Start collection for 30 seconds
            collection_task = asyncio.create_task(
                self.data_collector.start_streaming(
                    symbols=self.test_symbols[:2],  # Test with 2 symbols
                    update_interval=5.0
                )
            )

            # Wait for some data
            await asyncio.sleep(15)  # Collect for 15 seconds

            # Stop collection
            await self.data_collector.stop_streaming()

            # Validate results
            if len(collected_data) >= 2:  # Should have at least 2 data points
                logger.info(f"✅ Successfully collected {len(collected_data)} data points")

                # Check data quality
                for data_point in collected_data:
                    if not data_point.symbol or data_point.price <= 0:
                        logger.error("❌ Invalid data point detected")
                        return False

                return True
            else:
                logger.error(f"❌ Insufficient data collected: {len(collected_data)} points")
                return False

        except Exception as e:
            logger.error(f"Data collector test failed: {e}")
            return False

    async def test_websocket_streaming(self) -> bool:
        """Test WebSocket streaming service"""
        try:
            logger.info("🌐 Testing WebSocket Streaming Service...")

            # Initialize WebSocket service
            self.websocket_service = WebSocketStreamingService(
                ai_framework=self.ai_framework,
                port=8766  # Use different port for testing
            )

            # Start server in background
            server_task = asyncio.create_task(self.websocket_service.start_server())

            # Wait for server to start
            await asyncio.sleep(2)

            # Test client connection
            try:
                async with websockets.connect("ws://localhost:8766") as websocket:
                    # Test basic connection
                    welcome_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    welcome_data = json.loads(welcome_message)

                    if welcome_data.get('type') == 'connection_established':
                        logger.info("✅ WebSocket connection established")

                        # Test subscription
                        subscribe_message = {
                            'type': 'subscribe',
                            'symbols': ['RELIANCE.NS']
                        }
                        await websocket.send(json.dumps(subscribe_message))

                        # Wait for subscription confirmation
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        response_data = json.loads(response)

                        if response_data.get('type') == 'subscription_confirmed':
                            logger.info("✅ WebSocket subscription confirmed")

                            # Stop server
                            await self.websocket_service.stop_server()
                            return True
                        else:
                            logger.error(f"❌ Unexpected subscription response: {response_data}")
                            return False
                    else:
                        logger.error(f"❌ Unexpected welcome message: {welcome_data}")
                        return False

            except Exception as e:
                logger.error(f"WebSocket client test failed: {e}")
                return False
            finally:
                # Ensure server is stopped
                await self.websocket_service.stop_server()

        except Exception as e:
            logger.error(f"WebSocket service test failed: {e}")
            return False

    async def test_foundation_integration(self) -> bool:
        """Test integration with Foundation Week components"""
        try:
            logger.info("🏗️ Testing Foundation Week Integration...")

            # Test signal generator integration
            from src.ai_trading.ai_signal_generator import AISignalGenerator
            signal_generator = AISignalGenerator(self.ai_framework)

            # Create mock historical data
            import pandas as pd
            import numpy as np

            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            mock_data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, 100),
                'High': np.random.uniform(200, 250, 100),
                'Low': np.random.uniform(80, 150, 100),
                'Close': np.random.uniform(120, 220, 100),
                'Volume': np.random.randint(100000, 1000000, 100)
            }, index=dates)

            # Test signal generation
            signals = await signal_generator.generate_signals('RELIANCE.NS', mock_data)

            if signals and 'primary_signal' in signals:
                logger.info(f"✅ Signal generation successful: {signals.get('primary_signal')}")

                # Test fundamental analyzer integration
                from src.ai_trading.ai_fundamental_analyzer import AIFundamentalAnalyzer
                fundamental_analyzer = AIFundamentalAnalyzer(self.ai_framework)

                # Test fundamental analysis (may fail due to data limitations, that's ok)
                try:
                    fundamental_data = await fundamental_analyzer.analyze_stock('RELIANCE.NS')
                    if fundamental_data:
                        logger.info("✅ Fundamental analysis integration successful")
                    else:
                        logger.info("⚠️ Fundamental analysis returned no data (expected in test)")
                except Exception as fe:
                    logger.info(f"⚠️ Fundamental analysis test failed (expected): {fe}")

                # Test risk manager integration
                from src.ai_trading.ai_risk_manager import AIRiskManager
                risk_manager = AIRiskManager(self.ai_framework)

                risk_metrics = await risk_manager.calculate_position_size(
                    symbol='RELIANCE.NS',
                    signal='BUY',
                    confidence=0.8,
                    current_price=150.0,
                    portfolio_value=100000
                )

                if risk_metrics and 'position_size_percent' in risk_metrics:
                    logger.info("✅ Risk management integration successful")
                    return True
                else:
                    logger.error("❌ Risk management integration failed")
                    return False
            else:
                logger.error("❌ Signal generation integration failed")
                return False

        except Exception as e:
            logger.error(f"Foundation integration test failed: {e}")
            return False

    async def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end pipeline"""
        try:
            logger.info("🔄 Testing End-to-End Pipeline...")

            # Initialize complete system
            config = {
                'symbols': self.test_symbols[:2],  # Test with 2 symbols
                'websocket_port': 8767,  # Different port for testing
                'data_update_interval': 5.0
            }

            self.intelligence_system = RealTimeMarketIntelligence(config)

            # Start system in background (with timeout)
            system_task = asyncio.create_task(self.intelligence_system.start_system())

            # Wait for system to initialize
            await asyncio.sleep(5)

            # Test WebSocket connection to the complete system
            try:
                async with websockets.connect("ws://localhost:8767") as websocket:
                    # Subscribe to a symbol
                    subscribe_message = {
                        'type': 'subscribe',
                        'symbols': ['RELIANCE.NS'],
                        'subscription_level': 'premium'
                    }
                    await websocket.send(json.dumps(subscribe_message))

                    # Wait for data
                    received_messages = 0
                    for _ in range(10):  # Try to receive 10 messages
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                            data = json.loads(message)
                            received_messages += 1

                            if data.get('type') == 'market_data':
                                logger.info(
                                    f"📊 Received enhanced market data: {data.get('symbol')} @ ₹{data.get('price', 'N/A')}")
                        except asyncio.TimeoutError:
                            break

                    if received_messages >= 3:  # Should receive at least 3 messages
                        logger.info(f"✅ End-to-end pipeline successful: {received_messages} messages")
                        return True
                    else:
                        logger.error(f"❌ Insufficient messages received: {received_messages}")
                        return False

            except Exception as e:
                logger.error(f"End-to-end WebSocket test failed: {e}")
                return False
            finally:
                # Stop the system
                await self.intelligence_system.stop_system()

        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            return False

    async def test_performance_benchmarks(self) -> bool:
        """Test system performance benchmarks"""
        try:
            logger.info("⚡ Testing Performance Benchmarks...")

            # Test data processing speed
            if not self.data_collector:
                self.data_collector = RealTimeDataCollector(self.ai_framework)

            # Measure processing time for data collection
            processing_times = []

            def timed_data_handler(data_point: MarketDataPoint):
                processing_times.append(time.time())

            self.data_collector.add_subscriber(timed_data_handler)

            # Collect performance data
            start_time = time.time()
            collection_task = asyncio.create_task(
                self.data_collector.start_streaming(
                    symbols=['RELIANCE.NS'],
                    update_interval=2.0
                )
            )

            await asyncio.sleep(10)  # Collect for 10 seconds
            await self.data_collector.stop_streaming()

            total_time = time.time() - start_time

            # Calculate performance metrics
            if len(processing_times) >= 2:
                avg_interval = sum(processing_times[i] - processing_times[i - 1]
                                   for i in range(1, len(processing_times))) / (len(processing_times) - 1)

                logger.info(f"📊 Performance Metrics:")
                logger.info(f"   Total data points: {len(processing_times)}")
                logger.info(f"   Average interval: {avg_interval:.2f}s")
                logger.info(f"   Expected interval: 2.0s")

                # Performance criteria
                performance_ok = (
                        avg_interval <= 3.0 and  # Within 50% of expected interval
                        len(processing_times) >= 3  # At least 3 data points in 10 seconds
                )

                if performance_ok:
                    logger.info("✅ Performance benchmarks met")
                    return True
                else:
                    logger.error("❌ Performance benchmarks not met")
                    return False
            else:
                logger.error("❌ Insufficient performance data")
                return False

        except Exception as e:
            logger.error(f"Performance benchmark test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling and recovery mechanisms"""
        try:
            logger.info("🛡️ Testing Error Handling & Recovery...")

            # Test data collector error recovery
            data_collector = RealTimeDataCollector(ai_framework=None)  # No AI framework

            error_count = 0

            def error_tracking_handler(data_point: MarketDataPoint):
                nonlocal error_count
                if data_point.quality_score < 0.5:  # Consider low quality as error indicator
                    error_count += 1

            data_collector.add_subscriber(error_tracking_handler)

            # Test with invalid symbols (should handle gracefully)
            collection_task = asyncio.create_task(
                data_collector.start_streaming(
                    symbols=['INVALID.NS', 'RELIANCE.NS'],
                    update_interval=3.0
                )
            )

            await asyncio.sleep(10)  # Let it run and handle errors
            await data_collector.stop_streaming()

            # Check if system stayed operational despite errors
            metrics = data_collector.get_metrics()

            if metrics['data_points_processed'] > 0:
                logger.info("✅ System remained operational despite errors")
                logger.info(f"   Processed: {metrics['data_points_processed']} points")
                logger.info(f"   Error rate: {metrics['error_rate']:.1f}%")
                return True
            else:
                logger.error("❌ System failed to process any data")
                return False

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_analysis_quality(self) -> bool:
        """Test the quality of real-time analysis"""
        try:
            logger.info("🎯 Testing Real-Time Analysis Quality...")

            # Test with Foundation Week components
            from src.ai_trading.confluence_scoring_system import ConfluenceScorer
            confluence_scorer = ConfluenceScorer()

            # Create test analysis data
            test_cases = [
                {
                    'technical_signal': 'BUY',
                    'technical_confidence': 0.8,
                    'fundamental_score': 75,
                    'fundamental_recommendation': 'BUY',
                    'risk_score': 3,
                    'position_size': 5.0
                },
                {
                    'technical_signal': 'SELL',
                    'technical_confidence': 0.9,
                    'fundamental_score': 30,
                    'fundamental_recommendation': 'SELL',
                    'risk_score': 7,
                    'position_size': 2.0
                },
                {
                    'technical_signal': 'HOLD',
                    'technical_confidence': 0.5,
                    'fundamental_score': 50,
                    'fundamental_recommendation': 'HOLD',
                    'risk_score': 5,
                    'position_size': 3.0
                }
            ]

            analysis_results = []

            for i, test_case in enumerate(test_cases):
                try:
                    result = confluence_scorer.calculate_confluence_score(test_case)
                    analysis_results.append(result)
                    logger.info(f"   Test {i + 1}: {result.get('final_signal')} "
                                f"(Score: {result.get('confluence_score', 0):.0f})")
                except Exception as e:
                    logger.error(f"   Test {i + 1} failed: {e}")
                    analysis_results.append(None)

            # Validate analysis quality
            successful_analyses = sum(1 for result in analysis_results if result is not None)

            if successful_analyses >= 2:  # At least 2 out of 3 should work
                logger.info(f"✅ Analysis quality test passed: {successful_analyses}/3 successful")
                return True
            else:
                logger.error(f"❌ Analysis quality test failed: {successful_analyses}/3 successful")
                return False

        except Exception as e:
            logger.error(f"Analysis quality test failed: {e}")
            return False

    async def generate_validation_report(self, success_rate: float,
                                         passed_tests: int, total_tests: int):
        """Generate comprehensive validation report"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 PHASE 1 DAY 8 VALIDATION REPORT")
        logger.info("=" * 60)

        # Overall results
        grade = self._calculate_grade(success_rate)
        logger.info(f"🎯 Overall Grade: {grade}")
        logger.info(f"📊 Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests} tests passed)")

        # Detailed test results
        logger.info(f"\n📝 Detailed Test Results:")
        logger.info("-" * 40)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            duration = result.get('duration', 0)
            logger.info(f"{status} {test_name:<25} ({duration:.2f}s)")

            if 'error' in result:
                logger.info(f"    Error: {result['error']}")

        # Performance summary
        logger.info(f"\n⚡ Performance Summary:")
        logger.info("-" * 40)

        total_test_time = sum(r.get('duration', 0) for r in self.test_results.values())
        logger.info(f"Total Test Duration: {total_test_time:.1f} seconds")
        logger.info(f"Average Test Time: {total_test_time / total_tests:.1f} seconds")

        # Component status
        logger.info(f"\n🏗️ Component Status:")
        logger.info("-" * 40)

        components = {
            'AI Framework': 'ai_framework' in [k.lower().replace(' ', '_') for k in self.test_results.keys()],
            'Data Collector': any('data' in k.lower() for k in self.test_results.keys()),
            'WebSocket Service': any('websocket' in k.lower() for k in self.test_results.keys()),
            'Foundation Integration': any('foundation' in k.lower() for k in self.test_results.keys()),
            'End-to-End Pipeline': any('end' in k.lower() for k in self.test_results.keys())
        }

        for component, tested in components.items():
            status = "✅ TESTED" if tested else "⚠️ NOT TESTED"
            logger.info(f"{status} {component}")

        # Phase 1 Day 8 completion assessment
        logger.info(f"\n🏆 Phase 1 Day 8 Assessment:")
        logger.info("-" * 40)

        if success_rate >= 0.9:
            logger.info("🌟 EXCEPTIONAL - Real-time streaming architecture successfully implemented")
            logger.info("🚀 Ready for Phase 1 Day 9 - Advanced ML Integration")
        elif success_rate >= 0.8:
            logger.info("✅ SUCCESSFUL - Real-time pipeline operational with minor issues")
            logger.info("🔧 Address any failed tests before continuing")
        elif success_rate >= 0.7:
            logger.info("⚠️ PARTIAL SUCCESS - Core functionality working but needs improvement")
            logger.info("🛠️ Review and fix failed components")
        else:
            logger.info("❌ NEEDS WORK - Significant issues with real-time implementation")
            logger.info("🔄 Recommend revisiting architecture before proceeding")

        # Next steps
        logger.info(f"\n🎯 Recommended Next Steps:")
        logger.info("-" * 40)

        if success_rate >= 0.8:
            logger.info("1. 📄 Update context_summary.md with Phase 1 Day 8 completion")
            logger.info("2. 📝 Update changelog.md with real-time streaming features")
            logger.info("3. 🔄 Update requirements.txt (pip freeze > requirements.txt)")
            logger.info("4. 🚀 Begin Phase 1 Day 9: Advanced ML Model Integration")
            logger.info("5. 🎯 Focus on predictive analytics and ensemble learning")
        else:
            logger.info("1. 🔍 Review failed test cases and error logs")
            logger.info("2. 🛠️ Fix identified issues in real-time pipeline")
            logger.info("3. 🔄 Re-run validation tests")
            logger.info("4. 📚 Ensure Foundation Week components are working properly")
            logger.info("5. ⚡ Optimize performance where needed")

        logger.info("=" * 60)

        return grade

    def _calculate_grade(self, success_rate: float) -> str:
        """Calculate grade based on success rate"""
        if success_rate >= 0.95:
            return "A+ (EXCEPTIONAL - 95%+)"
        elif success_rate >= 0.9:
            return "A+ (OUTSTANDING - 90%+)"
        elif success_rate >= 0.85:
            return "A (EXCELLENT - 85%+)"
        elif success_rate >= 0.8:
            return "A- (VERY GOOD - 80%+)"
        elif success_rate >= 0.75:
            return "B+ (GOOD - 75%+)"
        elif success_rate >= 0.7:
            return "B (SATISFACTORY - 70%+)"
        else:
            return "C (NEEDS IMPROVEMENT - <70%)"


# Standalone test runner
async def main():
    """Main test runner function"""
    print("🧪 MarketPulse Phase 1 Day 8 - Real-Time Pipeline Validation")
    print("=" * 65)
    print("Testing integration of Foundation Week system with streaming architecture")
    print("Foundation Week Status: ✅ COMPLETE (Grade A+ Exceptional)")
    print("=" * 65)

    # Initialize validator
    validator = Phase1Day8Validator()

    try:
        # Run comprehensive validation
        success = await validator.run_comprehensive_tests()

        if success:
            print("\n🎉 PHASE 1 DAY 8 VALIDATION SUCCESSFUL!")
            print("🚀 Real-time streaming architecture is operational")
            print("📈 Ready to continue Phase 1 development")
        else:
            print("\n⚠️ PHASE 1 DAY 8 VALIDATION INCOMPLETE")
            print("🔧 Review failed tests and address issues")
            print("📋 Check validation report for detailed recommendations")

        return success

    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {e}")
        logger.error(f"Validation failed with error: {e}")
        return False


if __name__ == "__main__":
    # Run Phase 1 Day 8 validation
    success = asyncio.run(main())

    if success:
        print(f"\n✅ Phase 1 Day 8 validation completed successfully!")
        print(f"📊 Real-time data pipeline is ready for production use")
    else:
        print(f"\n❌ Phase 1 Day 8 validation needs attention")
        print(f"🛠️ Address failed tests before proceeding")

    exit(0 if success else 1)