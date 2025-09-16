# File: src/ai_trading/multi_timeframe_collector.py

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from pathlib import Path
import sys
import pandas as pd
import pandas_ta as ta

# Add project paths for AI framework
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "01_Framework_Core"))


class MultiTimeframeCollector:
    """
    Collects market data across multiple timeframes for confluence analysis.
    Supports: 1m, 5m, 15m, 1H, 4H, 1D timeframes
    """

    def __init__(self):
        self.timeframes = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }

        # Timeframe weights for confluence scoring (higher = more important)
        self.timeframe_weights = {
            '1m': 0.05,  # Lowest weight - too noisy
            '5m': 0.10,  # Low weight - short-term noise
            '15m': 0.15,  # Medium-low weight
            '1h': 0.20,  # Medium weight - good for entries
            '4h': 0.25,  # High weight - swing trading
            '1d': 0.25  # Highest weight - primary trend
        }

        print("🏗️ MultiTimeframeCollector initialized")
        print(f"📊 Supported timeframes: {list(self.timeframes.keys())}")
        print(f"⚖️ Timeframe weights: {self.timeframe_weights}")

    async def collect_symbol_data_parallel(self, symbol: str, days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all timeframes in parallel (much faster)
        """
        print(f"📈 Collecting data for {symbol} ({days_back} days) - PARALLEL")

        import asyncio
        import concurrent.futures

        def fetch_single_timeframe(tf_data):
            tf_name, tf_code = tf_data
            try:
                ticker = yf.Ticker(symbol)

                if tf_name in ['1m', '5m']:
                    period = min(7, days_back)
                    data = ticker.history(period=f"{period}d", interval=tf_code)
                else:
                    data = ticker.history(period=f"{days_back}d", interval=tf_code)

                if not data.empty:
                    data = self._add_basic_indicators(data, tf_name)
                    return tf_name, data
                else:
                    return tf_name, None

            except Exception as e:
                print(f"    ❌ {tf_name}: Error - {e}")
                return tf_name, None

        # Fetch all timeframes in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch_single_timeframe, (tf_name, tf_code))
                       for tf_name, tf_code in self.timeframes.items()]

            data_collection = {}
            for future in concurrent.futures.as_completed(futures):
                tf_name, data = future.result()
                if data is not None:
                    data_collection[tf_name] = data
                    print(f"    ✅ {tf_name}: {len(data)} data points")

        print(f"🎯 Collected data for {len(data_collection)} timeframes (PARALLEL)")
        return data_collection

    def _add_basic_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add professional-grade technical indicators
        """
        import pandas_ta as ta

        # Professional RSI (Wilder's method)
        data['RSI'] = ta.rsi(data['Close'], length=14)

        # Multiple SMAs
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)

        # MACD
        macd_data = ta.macd(data['Close'])
        data = pd.concat([data, macd_data], axis=1)

        # Bollinger Bands
        bb_data = ta.bbands(data['Close'], length=20)
        data = pd.concat([data, bb_data], axis=1)

        # Volume indicators
        data['Volume_SMA'] = ta.sma(data['Volume'], length=20)
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

        # Price analysis (keep your existing logic)
        data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20'] * 100
        data['Price_vs_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50'] * 100
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100

        # Fixed support/resistance (no lookahead bias)
        data['Local_High'] = data['High'].rolling(window=10).max().shift(1)
        data['Local_Low'] = data['Low'].rolling(window=10).min().shift(1)

        return data

    def get_latest_signals(self, data_collection: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Extract latest trading signals from each timeframe with NaN handling
        """
        signals = {}

        for timeframe, data in data_collection.items():
            if data.empty or len(data) < 2:
                continue

            latest = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else latest

            # Safely get values, handling NaN
            close_price = latest['Close']
            sma_20 = latest.get('SMA_20', None)
            sma_50 = latest.get('SMA_50', None)
            rsi_value = latest.get('RSI', None)

            # Determine trend direction with NaN checks
            trend_signal = "NEUTRAL"

            # Only compare if all values are valid (not NaN)
            if (pd.notna(close_price) and pd.notna(sma_20) and pd.notna(sma_50)):
                if close_price > sma_20 > sma_50:
                    trend_signal = "BULLISH"
                elif close_price < sma_20 < sma_50:
                    trend_signal = "BEARISH"

            # Momentum analysis with NaN check
            momentum_signal = "NEUTRAL"
            if pd.notna(rsi_value):
                if rsi_value > 70:
                    momentum_signal = "OVERBOUGHT"
                elif rsi_value < 30:
                    momentum_signal = "OVERSOLD"
                elif 40 <= rsi_value <= 60:
                    momentum_signal = "NEUTRAL"
                elif rsi_value > 50:
                    momentum_signal = "BULLISH"
                else:
                    momentum_signal = "BEARISH"

            # Volume confirmation with NaN check
            volume_ratio = latest.get('Volume_Ratio', 1.0)
            volume_signal = "NORMAL"
            if pd.notna(volume_ratio):
                if volume_ratio > 1.5:
                    volume_signal = "HIGH"
                elif volume_ratio < 0.5:
                    volume_signal = "LOW"

            # Price change with NaN check
            price_change_pct = latest.get('Price_Change_Pct', 0.0)
            if pd.isna(price_change_pct):
                price_change_pct = 0.0

            # Price vs SMA with NaN check
            price_vs_sma20 = latest.get('Price_vs_SMA20', 0.0)
            if pd.isna(price_vs_sma20):
                price_vs_sma20 = 0.0

            signals[timeframe] = {
                'price': close_price,
                'trend': trend_signal,
                'momentum': momentum_signal,
                'rsi': rsi_value if pd.notna(rsi_value) else 50.0,  # Default to neutral RSI
                'volume': volume_signal,
                'volume_ratio': volume_ratio if pd.notna(volume_ratio) else 1.0,
                'price_vs_sma20': price_vs_sma20,
                'price_change_pct': price_change_pct,
                'weight': self.timeframe_weights[timeframe]
            }

        return signals


# Test the collector
async def test_collector():
    print("🧪 Testing MultiTimeframe Collector")
    print("=" * 50)

    collector = MultiTimeframeCollector()
    symbol = "RELIANCE.NS"

    try:
        # FIXED: Use await and correct method name
        data = await collector.collect_symbol_data_parallel(symbol, days_back=10)
        signals = collector.get_latest_signals(data)

        print(f"\n🎯 Latest signals for {symbol}:")
        for timeframe, signal in signals.items():
            print(f"  {timeframe:>3s}: Trend={signal['trend']:>8s} | "
                  f"RSI={signal['rsi']:>6.1f} | Vol={signal['volume']:>6s} | "
                  f"Weight={signal['weight']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_collector())