# src/ai_trading/confluence_scoring_system.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class ConfluenceScore:
    """Structured confluence analysis result"""
    symbol: str
    overall_score: float  # 0-100
    signal_direction: str  # BUY, SELL, HOLD
    confidence_level: str  # HIGH, MEDIUM, LOW

    # Breakdown by category
    trend_score: float
    momentum_score: float
    volume_score: float
    support_resistance_score: float

    # Timeframe analysis
    timeframe_scores: Dict[str, float]
    timeframe_agreement: float  # % of timeframes agreeing

    # Supporting evidence
    bullish_factors: List[str]
    bearish_factors: List[str]
    neutral_factors: List[str]

    # Key levels
    key_resistance: float
    key_support: float
    current_price: float

    # Risk assessment
    volatility_risk: str  # LOW, MEDIUM, HIGH
    trend_strength: str  # WEAK, MODERATE, STRONG

    timestamp: datetime


class ConfluenceScoringSystem:
    """
    Advanced multi-timeframe confluence scoring system
    Analyzes technical signals across timeframes and indicators
    """

    def __init__(self):
        self.scoring_weights = {
            'trend': 0.30,  # 30% weight for trend analysis
            'momentum': 0.25,  # 25% weight for momentum
            'volume': 0.20,  # 20% weight for volume
            'support_resistance': 0.25  # 25% weight for S/R
        }

        self.timeframe_weights = {
            '1m': 0.05,  # Very short term
            '5m': 0.10,  # Short term
            '15m': 0.15,  # Short-medium term
            '1h': 0.20,  # Medium term
            '4h': 0.25,  # Medium-long term
            '1d': 0.25  # Long term
        }

        # Signal strength thresholds
        self.thresholds = {
            'strong_bullish': 75,
            'moderate_bullish': 60,
            'neutral': 40,
            'moderate_bearish': 25,
            'strong_bearish': 0
        }

    def analyze_confluence(self,
                           symbol: str,
                           technical_data: Dict[str, pd.DataFrame],
                           timeframe_signals: Dict[str, Dict]) -> ConfluenceScore:
        """
        Perform comprehensive confluence analysis
        """
        print(f"🔍 Analyzing confluence for {symbol}")

        current_price = self._get_current_price(technical_data)

        # 1. Analyze each category
        trend_analysis = self._analyze_trend_confluence(technical_data, timeframe_signals)
        momentum_analysis = self._analyze_momentum_confluence(technical_data, timeframe_signals)
        volume_analysis = self._analyze_volume_confluence(technical_data, timeframe_signals)
        sr_analysis = self._analyze_support_resistance_confluence(technical_data)

        # 2. Calculate weighted overall score
        overall_score = (
                trend_analysis['score'] * self.scoring_weights['trend'] +
                momentum_analysis['score'] * self.scoring_weights['momentum'] +
                volume_analysis['score'] * self.scoring_weights['volume'] +
                sr_analysis['score'] * self.scoring_weights['support_resistance']
        )

        # 3. Analyze timeframe agreement
        tf_scores = self._calculate_timeframe_scores(timeframe_signals)
        tf_agreement = self._calculate_timeframe_agreement(tf_scores)

        # 4. Determine signal direction and confidence
        signal_direction = self._determine_signal_direction(overall_score)
        confidence_level = self._determine_confidence_level(overall_score, tf_agreement)

        # 5. Compile factors
        bullish_factors = (trend_analysis['bullish'] + momentum_analysis['bullish'] +
                           volume_analysis['bullish'] + sr_analysis['bullish'])
        bearish_factors = (trend_analysis['bearish'] + momentum_analysis['bearish'] +
                           volume_analysis['bearish'] + sr_analysis['bearish'])
        neutral_factors = (trend_analysis['neutral'] + momentum_analysis['neutral'] +
                           volume_analysis['neutral'] + sr_analysis['neutral'])

        # 6. Key levels
        key_support, key_resistance = self._identify_key_levels(technical_data, current_price)

        # 7. Risk assessment
        volatility_risk = self._assess_volatility_risk(technical_data)
        trend_strength = self._assess_trend_strength(trend_analysis['score'])

        return ConfluenceScore(
            symbol=symbol,
            overall_score=round(overall_score, 1),
            signal_direction=signal_direction,
            confidence_level=confidence_level,
            trend_score=round(trend_analysis['score'], 1),
            momentum_score=round(momentum_analysis['score'], 1),
            volume_score=round(volume_analysis['score'], 1),
            support_resistance_score=round(sr_analysis['score'], 1),
            timeframe_scores=tf_scores,
            timeframe_agreement=round(tf_agreement, 1),
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            key_resistance=key_resistance,
            key_support=key_support,
            current_price=current_price,
            volatility_risk=volatility_risk,
            trend_strength=trend_strength,
            timestamp=datetime.now()
        )

    def _get_current_price(self, technical_data: Dict[str, pd.DataFrame]) -> float:
        """Get current price from the most recent data"""
        for tf_data in technical_data.values():
            if not tf_data.empty:
                return tf_data['Close'].iloc[-1]
        return 0.0

    def _analyze_trend_confluence(self,
                                  technical_data: Dict[str, pd.DataFrame],
                                  timeframe_signals: Dict[str, Dict]) -> Dict:
        """Analyze trend confluence across timeframes and indicators"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        score = 50  # Start neutral

        # Get primary timeframe data (longest available)
        primary_data = self._get_primary_timeframe_data(technical_data)

        if primary_data.empty:
            return {'score': 50, 'bullish': [], 'bearish': [], 'neutral': ['No data available']}

        latest = primary_data.iloc[-1]

        # 1. Moving Average Analysis
        ma_signals = self._analyze_moving_averages(primary_data)
        if ma_signals['bullish']:
            score += 10
            bullish_factors.extend(ma_signals['bullish'])
        if ma_signals['bearish']:
            score -= 10
            bearish_factors.extend(ma_signals['bearish'])

        # 2. ADX Trend Strength
        if 'ADX' in primary_data.columns:
            adx = latest.get('ADX', 25)
            if pd.notna(adx):
                if adx > 25:
                    score += 5
                    bullish_factors.append(f"Strong trend (ADX: {adx:.1f})")
                elif adx < 20:
                    score -= 5
                    neutral_factors.append(f"Weak trend (ADX: {adx:.1f})")

        # 3. Timeframe Trend Agreement
        bullish_tfs = sum(1 for tf_signal in timeframe_signals.values()
                          if tf_signal.get('trend') == 'BULLISH')
        bearish_tfs = sum(1 for tf_signal in timeframe_signals.values()
                          if tf_signal.get('trend') == 'BEARISH')

        if bullish_tfs > bearish_tfs:
            score += 5 * (bullish_tfs - bearish_tfs)
            bullish_factors.append(f"Bullish trend in {bullish_tfs}/{len(timeframe_signals)} timeframes")
        elif bearish_tfs > bullish_tfs:
            score -= 5 * (bearish_tfs - bullish_tfs)
            bearish_factors.append(f"Bearish trend in {bearish_tfs}/{len(timeframe_signals)} timeframes")

        return {
            'score': max(0, min(100, score)),
            'bullish': bullish_factors,
            'bearish': bearish_factors,
            'neutral': neutral_factors
        }

    def _analyze_momentum_confluence(self,
                                     technical_data: Dict[str, pd.DataFrame],
                                     timeframe_signals: Dict[str, Dict]) -> Dict:
        """Analyze momentum confluence across indicators"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        score = 50

        primary_data = self._get_primary_timeframe_data(technical_data)
        if primary_data.empty:
            return {'score': 50, 'bullish': [], 'bearish': [], 'neutral': ['No data available']}

        latest = primary_data.iloc[-1]

        # 1. RSI Analysis
        if 'RSI_14' in primary_data.columns:
            rsi = latest.get('RSI_14')
            if pd.notna(rsi):
                if rsi < 30:
                    score += 15
                    bullish_factors.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    score -= 15
                    bearish_factors.append(f"RSI overbought ({rsi:.1f})")
                elif 40 <= rsi <= 60:
                    neutral_factors.append(f"RSI neutral ({rsi:.1f})")

        # 2. MACD Analysis
        macd_cols = [col for col in primary_data.columns if 'MACD' in col and 'MACDs' not in col]
        signal_cols = [col for col in primary_data.columns if 'MACDs' in col]

        if macd_cols and signal_cols:
            macd = latest[macd_cols[0]]
            signal = latest[signal_cols[0]]
            if pd.notna(macd) and pd.notna(signal):
                if macd > signal and macd > 0:
                    score += 10
                    bullish_factors.append("MACD bullish above signal line")
                elif macd < signal and macd < 0:
                    score -= 10
                    bearish_factors.append("MACD bearish below signal line")

        # 3. Stochastic Analysis
        if 'STOCH_K' in primary_data.columns:
            stoch_k = latest.get('STOCH_K')
            if pd.notna(stoch_k):
                if stoch_k < 20:
                    score += 10
                    bullish_factors.append(f"Stochastic oversold ({stoch_k:.1f})")
                elif stoch_k > 80:
                    score -= 10
                    bearish_factors.append(f"Stochastic overbought ({stoch_k:.1f})")

        # 4. Multi-timeframe momentum agreement
        bullish_momentum = sum(1 for tf_signal in timeframe_signals.values()
                               if tf_signal.get('momentum') in ['BULLISH', 'OVERSOLD'])
        bearish_momentum = sum(1 for tf_signal in timeframe_signals.values()
                               if tf_signal.get('momentum') in ['BEARISH', 'OVERBOUGHT'])

        if bullish_momentum > bearish_momentum:
            score += 5
            bullish_factors.append(f"Bullish momentum in {bullish_momentum} timeframes")
        elif bearish_momentum > bullish_momentum:
            score -= 5
            bearish_factors.append(f"Bearish momentum in {bearish_momentum} timeframes")

        return {
            'score': max(0, min(100, score)),
            'bullish': bullish_factors,
            'bearish': bearish_factors,
            'neutral': neutral_factors
        }

    def _analyze_volume_confluence(self,
                                   technical_data: Dict[str, pd.DataFrame],
                                   timeframe_signals: Dict[str, Dict]) -> Dict:
        """Analyze volume confluence"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        score = 50

        primary_data = self._get_primary_timeframe_data(technical_data)
        if primary_data.empty:
            return {'score': 50, 'bullish': [], 'bearish': [], 'neutral': ['No volume data']}

        latest = primary_data.iloc[-1]

        # 1. Volume Ratio Analysis
        if 'VOLUME_RATIO' in primary_data.columns:
            vol_ratio = latest.get('VOLUME_RATIO', 1.0)
            if pd.notna(vol_ratio):
                if vol_ratio > 1.5:
                    score += 10
                    bullish_factors.append(f"High volume activity ({vol_ratio:.1f}x average)")
                elif vol_ratio < 0.5:
                    score -= 5
                    bearish_factors.append(f"Low volume activity ({vol_ratio:.1f}x average)")

        # 2. On Balance Volume
        if 'OBV' in primary_data.columns and len(primary_data) > 1:
            obv_current = latest.get('OBV')
            obv_previous = primary_data.iloc[-2].get('OBV')
            if pd.notna(obv_current) and pd.notna(obv_previous):
                obv_change = obv_current - obv_previous
                if obv_change > 0:
                    score += 5
                    bullish_factors.append("OBV trending up")
                elif obv_change < 0:
                    score -= 5
                    bearish_factors.append("OBV trending down")

        # 3. Chaikin Money Flow
        if 'CMF' in primary_data.columns:
            cmf = latest.get('CMF')
            if pd.notna(cmf):
                if cmf > 0.1:
                    score += 8
                    bullish_factors.append(f"Strong money flow (CMF: {cmf:.3f})")
                elif cmf < -0.1:
                    score -= 8
                    bearish_factors.append(f"Weak money flow (CMF: {cmf:.3f})")

        # 4. Multi-timeframe volume agreement
        high_volume_tfs = sum(1 for tf_signal in timeframe_signals.values()
                              if tf_signal.get('volume') == 'HIGH')

        if high_volume_tfs >= 2:
            score += 5
            bullish_factors.append(f"High volume in {high_volume_tfs} timeframes")

        return {
            'score': max(0, min(100, score)),
            'bullish': bullish_factors,
            'bearish': bearish_factors,
            'neutral': neutral_factors
        }

    def _analyze_support_resistance_confluence(self, technical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze support and resistance levels"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        score = 50

        primary_data = self._get_primary_timeframe_data(technical_data)
        if primary_data.empty:
            return {'score': 50, 'bullish': [], 'bearish': [], 'neutral': ['No S/R data']}

        latest = primary_data.iloc[-1]
        current_price = latest['Close']

        # 1. Bollinger Bands Position
        if 'BB_POSITION' in primary_data.columns:
            bb_pos = latest.get('BB_POSITION')
            if pd.notna(bb_pos):
                if bb_pos < 0.2:
                    score += 10
                    bullish_factors.append(f"Near BB lower band (oversold)")
                elif bb_pos > 0.8:
                    score -= 10
                    bearish_factors.append(f"Near BB upper band (overbought)")

        # 2. Pivot Point Analysis
        if all(col in primary_data.columns for col in ['PIVOT', 'R1', 'S1']):
            pivot = latest.get('PIVOT')
            r1 = latest.get('R1')
            s1 = latest.get('S1')

            if all(pd.notna(val) for val in [pivot, r1, s1]):
                if current_price > pivot:
                    score += 5
                    bullish_factors.append("Above pivot point")
                elif current_price < pivot:
                    score -= 5
                    bearish_factors.append("Below pivot point")

                # Distance to key levels
                resistance_distance = (r1 - current_price) / current_price * 100
                support_distance = (current_price - s1) / current_price * 100

                if resistance_distance > 2:  # Far from resistance
                    score += 3
                    bullish_factors.append("Far from resistance level")
                if support_distance > 2:  # Far from support
                    score += 3
                    bullish_factors.append("Good support cushion")

        # 3. SuperTrend Analysis
        if 'SUPERTREND_DIRECTION' in primary_data.columns:
            st_direction = latest.get('SUPERTREND_DIRECTION')
            if pd.notna(st_direction):
                if st_direction > 0:
                    score += 8
                    bullish_factors.append("SuperTrend bullish")
                else:
                    score -= 8
                    bearish_factors.append("SuperTrend bearish")

        return {
            'score': max(0, min(100, score)),
            'bullish': bullish_factors,
            'bearish': bearish_factors,
            'neutral': neutral_factors
        }

    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict:
        """Analyze moving average signals"""
        bullish = []
        bearish = []

        if data.empty or len(data) < 2:
            return {'bullish': [], 'bearish': []}

        latest = data.iloc[-1]
        current_price = latest['Close']

        # Short-term trend (SMA 20 vs 50)
        if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            if pd.notna(sma_20) and pd.notna(sma_50):
                if sma_20 > sma_50 and current_price > sma_20:
                    bullish.append("Short-term uptrend (SMA 20>50, price>SMA20)")
                elif sma_20 < sma_50 and current_price < sma_20:
                    bearish.append("Short-term downtrend (SMA 20<50, price<SMA20)")

        # Long-term trend (SMA 50 vs 200)
        if all(col in data.columns for col in ['SMA_50', 'SMA_200']):
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            if pd.notna(sma_50) and pd.notna(sma_200):
                if sma_50 > sma_200:
                    bullish.append("Long-term bullish (SMA 50>200)")
                elif sma_50 < sma_200:
                    bearish.append("Long-term bearish (SMA 50<200)")

        # EMA crossover
        if all(col in data.columns for col in ['EMA_12', 'EMA_26']):
            ema_12 = latest['EMA_12']
            ema_26 = latest['EMA_26']
            if pd.notna(ema_12) and pd.notna(ema_26):
                if ema_12 > ema_26:
                    bullish.append("EMA bullish crossover (12>26)")
                else:
                    bearish.append("EMA bearish crossover (12<26)")

        return {'bullish': bullish, 'bearish': bearish}

    def _get_primary_timeframe_data(self, technical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get the primary timeframe data (highest quality/longest)"""
        # Priority: 1d > 4h > 1h > others
        priority_order = ['1d', '4h', '1h', '15m', '5m', '1m']

        for timeframe in priority_order:
            if timeframe in technical_data and not technical_data[timeframe].empty:
                return technical_data[timeframe]

        # Fallback to any available data
        for data in technical_data.values():
            if not data.empty:
                return data

        return pd.DataFrame()

    def _calculate_timeframe_scores(self, timeframe_signals: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate individual timeframe scores"""
        tf_scores = {}

        for timeframe, signals in timeframe_signals.items():
            score = 50  # Start neutral

            trend = signals.get('trend', 'NEUTRAL')
            momentum = signals.get('momentum', 'NEUTRAL')
            volume = signals.get('volume', 'NORMAL')

            # Trend scoring
            if trend == 'BULLISH':
                score += 15
            elif trend == 'BEARISH':
                score -= 15

            # Momentum scoring
            if momentum == 'BULLISH' or momentum == 'OVERSOLD':
                score += 10
            elif momentum == 'BEARISH' or momentum == 'OVERBOUGHT':
                score -= 10

            # Volume scoring
            if volume == 'HIGH':
                score += 5
            elif volume == 'LOW':
                score -= 3

            tf_scores[timeframe] = max(0, min(100, score))

        return tf_scores

    def _calculate_timeframe_agreement(self, tf_scores: Dict[str, float]) -> float:
        """Calculate percentage of timeframes agreeing on direction"""
        if not tf_scores:
            return 0

        bullish_count = sum(1 for score in tf_scores.values() if score > 60)
        bearish_count = sum(1 for score in tf_scores.values() if score < 40)
        total_count = len(tf_scores)

        max_agreement = max(bullish_count, bearish_count)
        return (max_agreement / total_count) * 100

    def _determine_signal_direction(self, overall_score: float) -> str:
        """Determine signal direction from overall score"""
        if overall_score >= 60:
            return "BUY"
        elif overall_score <= 40:
            return "SELL"
        else:
            return "HOLD"

    def _determine_confidence_level(self, overall_score: float, tf_agreement: float) -> str:
        """Determine confidence level"""
        # High confidence: Strong score + high timeframe agreement
        if ((overall_score >= 75 or overall_score <= 25) and tf_agreement >= 70):
            return "HIGH"
        # Medium confidence: Moderate score or moderate agreement
        elif ((overall_score >= 65 or overall_score <= 35) and tf_agreement >= 50):
            return "MEDIUM"
        else:
            return "LOW"

    def _identify_key_levels(self, technical_data: Dict[str, pd.DataFrame], current_price: float) -> Tuple[
        float, float]:
        """Identify key support and resistance levels"""
        primary_data = self._get_primary_timeframe_data(technical_data)

        if primary_data.empty:
            return current_price * 0.98, current_price * 1.02

        latest = primary_data.iloc[-1]

        # Try to get calculated levels
        support = latest.get('S1', current_price * 0.98)
        resistance = latest.get('R1', current_price * 1.02)

        # Fallback calculations if NaN
        if pd.isna(support):
            support = current_price * 0.98
        if pd.isna(resistance):
            resistance = current_price * 1.02

        return float(support), float(resistance)

    def _assess_volatility_risk(self, technical_data: Dict[str, pd.DataFrame]) -> str:
        """Assess volatility risk level"""
        primary_data = self._get_primary_timeframe_data(technical_data)

        if primary_data.empty or 'VOLATILITY_INDEX' not in primary_data.columns:
            return "MEDIUM"

        vol_index = primary_data['VOLATILITY_INDEX'].iloc[-1]

        if pd.isna(vol_index):
            return "MEDIUM"
        elif vol_index > 70:
            return "HIGH"
        elif vol_index < 30:
            return "LOW"
        else:
            return "MEDIUM"

    def _assess_trend_strength(self, trend_score: float) -> str:
        """Assess trend strength from trend score"""
        if trend_score >= 70 or trend_score <= 30:
            return "STRONG"
        elif trend_score >= 60 or trend_score <= 40:
            return "MODERATE"
        else:
            return "WEAK"

    def format_confluence_report(self, confluence: ConfluenceScore) -> str:
        """Format confluence analysis into readable report"""

        report = f"""
🔍 MULTI-TIMEFRAME CONFLUENCE ANALYSIS
{'=' * 60}
📊 Symbol: {confluence.symbol}
⏰ Analysis Time: {confluence.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
💰 Current Price: ₹{confluence.current_price:.2f}

🎯 OVERALL ASSESSMENT:
Signal: {confluence.signal_direction} ({confluence.confidence_level} confidence)
Confluence Score: {confluence.overall_score}/100
Timeframe Agreement: {confluence.timeframe_agreement:.1f}%

📈 COMPONENT ANALYSIS:
Trend Score: {confluence.trend_score}/100 ({confluence.trend_strength})
Momentum Score: {confluence.momentum_score}/100
Volume Score: {confluence.volume_score}/100
Support/Resistance Score: {confluence.support_resistance_score}/100

📊 TIMEFRAME BREAKDOWN:
{self._format_timeframe_scores(confluence.timeframe_scores)}

🔍 KEY LEVELS:
Support: ₹{confluence.key_support:.2f}
Resistance: ₹{confluence.key_resistance:.2f}
Volatility Risk: {confluence.volatility_risk}

✅ BULLISH FACTORS ({len(confluence.bullish_factors)}):
{self._format_factor_list(confluence.bullish_factors)}

❌ BEARISH FACTORS ({len(confluence.bearish_factors)}):
{self._format_factor_list(confluence.bearish_factors)}

⚪ NEUTRAL FACTORS ({len(confluence.neutral_factors)}):
{self._format_factor_list(confluence.neutral_factors)}
        """

        return report.strip()

    def _format_timeframe_scores(self, tf_scores: Dict[str, float]) -> str:
        """Format timeframe scores for display"""
        if not tf_scores:
            return "No timeframe data"

        formatted = []
        for tf, score in tf_scores.items():
            direction = "BUY" if score > 60 else "SELL" if score < 40 else "HOLD"
            formatted.append(f"{tf:>3s}: {score:>5.1f}/100 ({direction})")

        return "\n".join(formatted)

    def _format_factor_list(self, factors: List[str]) -> str:
        """Format factor list for display"""
        if not factors:
            return "None"

        return "\n".join([f"• {factor}" for factor in factors])


# Testing function
async def test_confluence_scoring():
    """Test the confluence scoring system"""
    print("🧪 Testing Multi-Timeframe Confluence Scoring System")
    print("=" * 60)

    # Import required modules for testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    try:
        from professional_technical_analyzer import ProfessionalTechnicalAnalyzer
        from multi_timeframe_collector import MultiTimeframeCollector
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    # Initialize components
    scorer = ConfluenceScoringSystem()
    analyzer = ProfessionalTechnicalAnalyzer()
    collector = MultiTimeframeCollector()

    symbol = "RELIANCE.NS"
    print(f"📊 Testing confluence analysis for {symbol}")

    try:
        # Collect multi-timeframe data
        data_collection = await collector.collect_symbol_data_parallel(symbol)
        if not data_collection:
            print("❌ No data collected")
            return

        # Add technical indicators to each timeframe
        technical_data = {}
        for timeframe, data in data_collection.items():
            technical_data[timeframe] = analyzer.add_all_indicators(data.copy())

        # Get timeframe signals
        tf_signals = collector.get_latest_signals(data_collection)

        # Perform confluence analysis
        confluence = scorer.analyze_confluence(symbol, technical_data, tf_signals)

        print("✅ Confluence analysis completed!")
        print("\n" + "=" * 60)
        print("CONFLUENCE ANALYSIS REPORT:")
        print("=" * 60)
        print(scorer.format_confluence_report(confluence))
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_confluence_scoring())