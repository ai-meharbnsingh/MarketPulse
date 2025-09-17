# test/test_realtime_system.py
"""
Test suite for real-time system components
"""

import unittest
import asyncio
from datetime import datetime


class TestRealtimeSystem(unittest.TestCase):
    """Test cases for real-time system"""

    def test_system_initialization(self):
        """Test system initialization"""
        from integration.phase1_day8_pipeline import RealtimeMarketIntelligenceSystem
        system = RealtimeMarketIntelligenceSystem()
        self.assertTrue(system.is_initialized)
        self.assertEqual(system.status, "READY")

    def test_data_collector(self):
        """Test real-time data collector"""
        from ai_trading.realtime_data_collector import RealtimeDataCollector
        collector = RealtimeDataCollector()
        self.assertFalse(collector.is_running)
        self.assertEqual(len(collector.subscribers), 0)

    def test_websocket_service(self):
        """Test WebSocket streaming service"""
        from ai_trading.websocket_streaming_service import WebSocketStreamingService
        service = WebSocketStreamingService()
        self.assertFalse(service.is_running)
        self.assertEqual(len(service.connected_clients), 0)

    async def test_integration_pipeline(self):
        """Test integration pipeline"""
        from integration.phase1_day8_pipeline import RealtimeMarketIntelligenceSystem
        system = RealtimeMarketIntelligenceSystem()

        status = system.get_system_status()
        self.assertIn('status', status)
        self.assertIn('components', status)


if __name__ == '__main__':
    unittest.main()