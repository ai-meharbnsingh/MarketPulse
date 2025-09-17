# src/ai_trading/realtime_data_collector.py
"""
Real-time market data collection system
"""

import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Callable, Optional


class RealtimeDataCollector:
    """Collect real-time market data from multiple sources"""

    def __init__(self):
        self.subscribers = []
        self.is_running = False
        self.collected_data = {}

    async def start_collection(self, symbols: List[str]):
        """Start real-time data collection"""
        self.is_running = True
        print(f"🚀 Starting real-time collection for {len(symbols)} symbols")

        # Simulate real-time data collection
        while self.is_running:
            for symbol in symbols:
                # Generate mock real-time data
                mock_data = {
                    'symbol': symbol,
                    'price': 100.0 + (hash(symbol) % 50),
                    'volume': 1000000,
                    'timestamp': datetime.now(),
                    'bid': 99.95,
                    'ask': 100.05
                }

                self.collected_data[symbol] = mock_data

                # Notify subscribers
                for callback in self.subscribers:
                    await callback(symbol, mock_data)

            await asyncio.sleep(1)  # Update every second

    def stop_collection(self):
        """Stop real-time data collection"""
        self.is_running = False
        print("⏹️ Stopped real-time data collection")

    def subscribe(self, callback: Callable):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest data for a symbol"""
        return self.collected_data.get(symbol)