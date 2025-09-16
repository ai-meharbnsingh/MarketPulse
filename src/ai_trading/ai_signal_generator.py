# src/ai_trading/enhanced_ai_signal_generator.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import sys
import os

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our modules
try:
    from professional_technical_analyzer import ProfessionalTechnicalAnalyzer, TechnicalSignal
    from multi_timeframe_collector import MultiTimeframeCollector
    from confluence_scoring_system import ConfluenceScoringSystem, ConfluenceScore
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    sys.exit(1)


@dataclass
class EnhancedAITradingSignal:
    """Enhanced trading signal with confluence analysis and proper risk management"""
    # Basic signal info
    symbol: str
    timeframe: str
    timestamp: datetime
    signal_id: str

    # Core signal
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    strength: str  # WEAK, MODERATE, STRONG

    # Price and risk management (based on current price)
    current_price: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float

    # Risk metrics
    risk_reward_ratio: float
    position_size_pct: float
    max_risk_pct: float
    risk_per_share: float

    # Confluence analysis
    confluence_score: float
    timeframe_agreement: float
    trend_strength: str
    volatility_risk: str

    # Supporting evidence
    bullish_factors: List[str]
    bearish_factors: List[str]
    key_indicators: List[str]
    warning_signs: List[str]

    # Key levels
    key_support: float
    key_resistance: float

    # AI analysis
    ai_reasoning: str
    ai_provider: str
    processing_time_ms: float


class MockAIEngine:
    """Enhanced mock AI engine"""

    def __init__(self):
        self.provider = "MockAI_Enhanced"

    async def get_completion(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0.1)

        # Extract confluence score from prompt for more realistic responses
        confluence_score = 65  # Default
        if "Confluence Score:" in prompt:
            try:
                line = [l for l in prompt.split('\n') if 'Confluence Score:' in l][0]
                confluence_score = float(line.split(':')[1].split('/')[0].strip())
            except:
                pass

        if confluence_score >= 70:
            return self._generate_strong_buy_signal()
        elif confluence_score >= 55:
            return self._generate_moderate_buy_signal()
        elif confluence_score <= 30:
            return self._generate_sell_signal()
        elif confluence_score <= 45:
            return self._generate_moderate_sell_signal()
        else:
            return self._generate_hold_signal()

    def _generate_strong_buy_signal(self) -> str:
        return """
        {
            "action": "BUY",
            "confidence": 85,
            "reasoning": "Strong confluence across multiple timeframes with bullish momentum indicators. Technical breakout confirmed with volume support. Multiple indicators aligning for upward move with good risk-reward setup.",
            "time_horizon": "5-10 days",
            "conviction_level": "HIGH"
        }
        """

    def _generate_moderate_buy_signal(self) -> str:
        return """
        {
            "action": "BUY", 
            "confidence": 65,
            "reasoning": "Moderate bullish confluence with some supporting technical indicators. Trend showing signs of strength but mixed signals suggest cautious optimism. Entry near support levels provides good risk management.",
            "time_horizon": "3-7 days",
            "conviction_level": "MEDIUM"
        }
        """

    def _generate_sell_signal(self) -> str:
        return """
        {
            "action": "SELL",
            "confidence": 75,
            "reasoning": "Bearish confluence with multiple timeframes showing weakness. Technical indicators suggesting downward pressure with volume confirmation. Risk-reward favors short position.",
            "time_horizon": "3-7 days", 
            "conviction_level": "MEDIUM"
        }
        """

    def _generate_moderate_sell_signal(self) -> str:
        return """
        {
            "action": "SELL",
            "confidence": 60,
            "reasoning": "Moderate bearish signals with some technical weakness showing. Mixed timeframe signals but overall bias to the downside. Conservative position sizing recommended.",
            "time_horizon": "2-5 days",
            "conviction_level": "MEDIUM"
        }
        """

    def _generate_hold_signal(self) -> str:
        return """
        {
            "action": "HOLD",
            "confidence": 45,
            "reasoning": "Mixed signals with no clear directional bias. Confluence analysis shows neutral stance across timeframes. Wait for clearer setup before taking position.",
            "time_horizon": "Wait for clarity",
            "conviction_level": "LOW"
        }
        """


class EnhancedAISignalGenerator:
    """
    Enhanced AI Signal Generator with proper confluence analysis and risk management
    """

    def __init__(self):
        self.technical_analyzer = ProfessionalTechnicalAnalyzer()
        self.timeframe_collector = MultiTimeframeCollector()
        self.confluence_scorer = ConfluenceScoringSystem()

        # Initialize mock AI engine
        self.ai_engine = MockAIEngine()

        # Trading style configurations
        self.trading_styles = {
            'scalp': {
                'risk_per_trade': 0.5,
                'atr_stop_multiplier': 1.0,
                'atr_target_multiplier': 2.0,
                'max_holding_period': '1 hour'
            },
            'day_trade': {
                'risk_per_trade': 1.0,
                'atr_stop_multiplier': 1.5,
                'atr_target_multiplier': 2.5,
                'max_holding_period': '1 day'
            },
            'swing_trade': {
                'risk_per_trade': 2.0,
                'atr_stop_multiplier': 2.0,
                'atr_target_multiplier': 3.0,
                'max_holding_period': '1 week'
            },
            'long_term': {
                'risk_per_trade': 3.0,
                'atr_stop_multiplier': 3.0,
                'atr_target_multiplier': 4.0,
                'max_holding_period': '3 months'
            }
        }

    async def generate_enhanced_signal(self,
                                       symbol: str,
                                       trading_style: str = 'swing_trade') -> EnhancedAITradingSignal:
        """
        Generate enhanced AI trading signal with proper confluence analysis
        """
        start_time = datetime.now()
        signal_id = f"{symbol}_{trading_style}_{int(start_time.timestamp())}"

        print(f"🚀 Generating enhanced AI signal for {symbol} ({trading_style})")

        try:
            # Step 1: Multi-timeframe data collection
            print("📊 Collecting multi-timeframe data...")
            data_collection = await self.timeframe_collector.collect_symbol_data_parallel(symbol)

            if not data_collection:
                raise Exception("No market data available")

            # Step 2: Add technical indicators to all timeframes
            print("🔧 Adding technical indicators across timeframes...")
            technical_data = {}
            for timeframe, data in data_collection.items():
                technical_data[timeframe] = self.technical_analyzer.add_all_indicators(data.copy())

            # Step 3: Get timeframe signals
            tf_signals = self.timeframe_collector.get_latest_signals(data_collection)

            # Step 4: Perform confluence analysis
            print("🔍 Performing confluence analysis...")
            confluence = self.confluence_scorer.analyze_confluence(symbol, technical_data, tf_signals)

            # Step 5: Create AI analysis prompt with confluence data
            ai_prompt = self._create_enhanced_prompt(symbol, trading_style, confluence, technical_data)

            # Step 6: Get AI analysis
            print("🤖 Getting AI analysis...")
            ai_response = await self.ai_engine.get_completion(ai_prompt)
            ai_analysis = self._parse_ai_response(ai_response)

            # Step 7: Calculate proper risk management based on current price
            current_price = confluence.current_price
            primary_data = self._get_primary_data(technical_data)
            atr = self._calculate_atr(primary_data)

            risk_params = self._calculate_enhanced_risk_parameters(
                current_price=current_price,
                atr=atr,
                confluence=confluence,
                ai_analysis=ai_analysis,
                trading_style=trading_style
            )

            # Step 8: Create enhanced signal
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            signal = EnhancedAITradingSignal(
                symbol=symbol,
                timeframe=f"Multi-TF Analysis",
                timestamp=start_time,
                signal_id=signal_id,
                action=ai_analysis.get('action', confluence.signal_direction),
                confidence=ai_analysis.get('confidence', confluence.overall_score),
                strength=self._determine_signal_strength(ai_analysis.get('confidence', confluence.overall_score)),
                current_price=current_price,
                entry_price=risk_params['entry_price'],
                stop_loss=risk_params['stop_loss'],
                target_1=risk_params['target_1'],
                target_2=risk_params['target_2'],
                target_3=risk_params['target_3'],
                risk_reward_ratio=risk_params['risk_reward_ratio'],
                position_size_pct=risk_params['position_size_pct'],
                max_risk_pct=self.trading_styles[trading_style]['risk_per_trade'],
                risk_per_share=risk_params['risk_per_share'],
                confluence_score=confluence.overall_score,
                timeframe_agreement=confluence.timeframe_agreement,
                trend_strength=confluence.trend_strength,
                volatility_risk=confluence.volatility_risk,
                bullish_factors=confluence.bullish_factors,
                bearish_factors=confluence.bearish_factors,
                key_indicators=self._extract_key_indicators(confluence),
                warning_signs=self._extract_warnings(confluence, ai_analysis),
                key_support=confluence.key_support,
                key_resistance=confluence.key_resistance,
                ai_reasoning=ai_analysis.get('reasoning', 'Analysis based on confluence scoring'),
                ai_provider=self.ai_engine.provider,
                processing_time_ms=processing_time
            )

            print(f"✅ Generated {signal.action} signal with {signal.confidence:.1f}% confidence")
            print(f"   Confluence: {signal.confluence_score:.1f}/100, TF Agreement: {signal.timeframe_agreement:.1f}%")

            return signal

        except Exception as e:
            print(f"❌ Error generating enhanced signal: {e}")
            return self._create_safe_signal(symbol, trading_style, str(e), signal_id)

    def _create_enhanced_prompt(self,
                                symbol: str,
                                trading_style: str,
                                confluence: ConfluenceScore,
                                technical_data: Dict[str, pd.DataFrame]) -> str:
        """Create enhanced AI prompt with confluence analysis"""

        return f"""
You are an expert quantitative trader analyzing {symbol} with comprehensive confluence analysis.

**TRADING CONTEXT:**
- Symbol: {symbol}
- Trading Style: {trading_style} 
- Risk per Trade: {self.trading_styles[trading_style]['risk_per_trade']}%
- Max Holding: {self.trading_styles[trading_style]['max_holding_period']}

**CONFLUENCE ANALYSIS RESULTS:**
- Current Price: ₹{confluence.current_price:.2f}
- Confluence Score: {confluence.overall_score:.1f}/100
- Signal Direction: {confluence.signal_direction} ({confluence.confidence_level} confidence)
- Timeframe Agreement: {confluence.timeframe_agreement:.1f}%

**COMPONENT SCORES:**
- Trend Strength: {confluence.trend_score:.1f}/100 ({confluence.trend_strength})
- Momentum: {confluence.momentum_score:.1f}/100  
- Volume: {confluence.volume_score:.1f}/100
- Support/Resistance: {confluence.support_resistance_score:.1f}/100

**KEY LEVELS:**
- Support: ₹{confluence.key_support:.2f}
- Resistance: ₹{confluence.key_resistance:.2f}
- Volatility Risk: {confluence.volatility_risk}

**BULLISH FACTORS ({len(confluence.bullish_factors)}):**
{self._format_factors(confluence.bullish_factors)}

**BEARISH FACTORS ({len(confluence.bearish_factors)}):**
{self._format_factors(confluence.bearish_factors)}

**TIMEFRAME SCORES:**
{self._format_tf_scores(confluence.timeframe_scores)}

Based on this comprehensive confluence analysis, provide your trading recommendation in JSON format:

{{
    "action": "BUY|SELL|HOLD",
    "confidence": 0-100,
    "reasoning": "Your detailed analysis incorporating the confluence data",
    "time_horizon": "expected holding period",
    "conviction_level": "HIGH|MEDIUM|LOW"
}}

Consider:
1. The confluence score and timeframe agreement levels
2. The balance of bullish vs bearish factors  
3. Current position relative to key support/resistance
4. The trading style timeframe and risk parameters
5. Overall market regime and volatility

Provide your professional recommendation.
        """

    def _format_factors(self, factors: List[str]) -> str:
        """Format factors for prompt"""
        if not factors:
            return "None"
        return "\n".join([f"- {factor}" for factor in factors[:5]])  # Limit to top 5

    def _format_tf_scores(self, tf_scores: Dict[str, float]) -> str:
        """Format timeframe scores for prompt"""
        formatted = []
        for tf, score in tf_scores.items():
            direction = "BUY" if score > 60 else "SELL" if score < 40 else "HOLD"
            formatted.append(f"- {tf}: {score:.1f}/100 ({direction})")
        return "\n".join(formatted)

    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response"""
        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except:
            # Fallback
            return {
                'action': 'HOLD',
                'confidence': 50,
                'reasoning': 'Failed to parse AI response',
                'conviction_level': 'LOW'
            }

    def _get_primary_data(self, technical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get primary timeframe data"""
        for tf in ['1d', '4h', '1h', '15m']:
            if tf in technical_data and not technical_data[tf].empty:
                return technical_data[tf]

        for data in technical_data.values():
            if not data.empty:
                return data

        return pd.DataFrame()

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR for risk management"""
        if data.empty or 'ATR' not in data.columns:
            if not data.empty:
                # Calculate simple ATR as fallback
                high_low = data['High'] - data['Low']
                return high_low.rolling(14).mean().iloc[-1] if len(data) >= 14 else high_low.mean()
            return 0.02  # 2% fallback

        atr = data['ATR'].iloc[-1]
        return atr if pd.notna(atr) else data['Close'].iloc[-1] * 0.02

    def _calculate_enhanced_risk_parameters(self,
                                            current_price: float,
                                            atr: float,
                                            confluence: ConfluenceScore,
                                            ai_analysis: Dict,
                                            trading_style: str) -> Dict:
        """Calculate enhanced risk parameters based on current price"""

        action = ai_analysis.get('action', confluence.signal_direction)
        confidence = ai_analysis.get('confidence', confluence.overall_score)

        if action == 'HOLD':
            return {
                'entry_price': current_price,
                'stop_loss': current_price,
                'target_1': current_price,
                'target_2': current_price,
                'target_3': current_price,
                'risk_reward_ratio': 0,
                'position_size_pct': 0,
                'risk_per_share': 0
            }

        # Get trading style parameters
        style_params = self.trading_styles[trading_style]
        stop_multiplier = style_params['atr_stop_multiplier']
        target_multiplier = style_params['atr_target_multiplier']

        # Calculate stop loss based on ATR and key levels
        if action == 'BUY':
            # For buy signals, stop loss below support or ATR-based
            atr_stop = current_price - (atr * stop_multiplier)
            support_stop = confluence.key_support * 0.995  # 0.5% below support
            stop_loss = max(atr_stop, support_stop)  # Use the higher (less risk)

            # Targets based on resistance and ATR
            atr_target1 = current_price + (atr * target_multiplier)
            resistance_target = confluence.key_resistance * 0.995  # Just below resistance
            target_1 = min(atr_target1, resistance_target)  # Use the closer one
            target_2 = target_1 * 1.01  # 1% above target 1
            target_3 = target_1 * 1.02  # 2% above target 1

        else:  # SELL
            # For sell signals, stop loss above resistance or ATR-based
            atr_stop = current_price + (atr * stop_multiplier)
            resistance_stop = confluence.key_resistance * 1.005  # 0.5% above resistance
            stop_loss = min(atr_stop, resistance_stop)  # Use the lower (less risk)

            # Targets based on support and ATR
            atr_target1 = current_price - (atr * target_multiplier)
            support_target = confluence.key_support * 1.005  # Just above support
            target_1 = max(atr_target1, support_target)  # Use the closer one
            target_2 = target_1 * 0.99  # 1% below target 1
            target_3 = target_1 * 0.98  # 2% below target 1

        # Calculate risk per share and position sizing
        risk_per_share = abs(current_price - stop_loss)
        base_risk_pct = style_params['risk_per_trade']

        # Adjust position size based on confidence and confluence
        confidence_multiplier = confidence / 100
        confluence_multiplier = confluence.overall_score / 100
        agreement_multiplier = confluence.timeframe_agreement / 100

        # Combined multiplier (conservative approach)
        combined_multiplier = (confidence_multiplier + confluence_multiplier + agreement_multiplier) / 3
        position_size_pct = base_risk_pct * combined_multiplier * 0.5  # 50% of calculated size for safety

        # Calculate risk-reward ratio
        reward_per_share = abs(target_1 - current_price)
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        return {
            'entry_price': current_price,
            'stop_loss': round(stop_loss, 2),
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'target_3': round(target_3, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_size_pct': round(position_size_pct, 2),
            'risk_per_share': round(risk_per_share, 2)
        }

    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength"""
        if confidence >= 80:
            return "STRONG"
        elif confidence >= 60:
            return "MODERATE"
        else:
            return "WEAK"

    def _extract_key_indicators(self, confluence: ConfluenceScore) -> List[str]:
        """Extract key indicators from confluence analysis"""
        indicators = []

        # Add trend indicators
        if confluence.trend_score >= 70:
            indicators.append("Bullish_Trend")
        elif confluence.trend_score <= 30:
            indicators.append("Bearish_Trend")

        # Add momentum indicators
        if confluence.momentum_score >= 70:
            indicators.append("Strong_Momentum")
        elif confluence.momentum_score <= 30:
            indicators.append("Weak_Momentum")

        # Add volume indicators
        if confluence.volume_score >= 70:
            indicators.append("Volume_Support")

        return indicators

    def _extract_warnings(self, confluence: ConfluenceScore, ai_analysis: Dict) -> List[str]:
        """Extract warnings from analysis"""
        warnings = []

        if confluence.volatility_risk == "HIGH":
            warnings.append("High volatility risk")

        if confluence.timeframe_agreement < 50:
            warnings.append("Low timeframe agreement")

        if confluence.confidence_level == "LOW":
            warnings.append("Low confidence setup")

        if len(confluence.bearish_factors) >= len(confluence.bullish_factors):
            warnings.append("Significant bearish factors present")

        return warnings

    def _create_safe_signal(self, symbol: str, trading_style: str, error: str,
                            signal_id: str) -> EnhancedAITradingSignal:
        """Create safe HOLD signal on error"""
        return EnhancedAITradingSignal(
            symbol=symbol,
            timeframe="ERROR",
            timestamp=datetime.now(),
            signal_id=signal_id,
            action="HOLD",
            confidence=0,
            strength="WEAK",
            current_price=0,
            entry_price=0,
            stop_loss=0,
            target_1=0,
            target_2=0,
            target_3=0,
            risk_reward_ratio=0,
            position_size_pct=0,
            max_risk_pct=0,
            risk_per_share=0,
            confluence_score=0,
            timeframe_agreement=0,
            trend_strength="UNKNOWN",
            volatility_risk="UNKNOWN",
            bullish_factors=[],
            bearish_factors=[],
            key_indicators=[],
            warning_signs=[error],
            key_support=0,
            key_resistance=0,
            ai_reasoning=f"Error in analysis: {error}",
            ai_provider="error",
            processing_time_ms=0
        )

    def format_enhanced_report(self, signal: EnhancedAITradingSignal) -> str:
        """Format enhanced signal report"""

        return f"""
🚀 ENHANCED AI TRADING SIGNAL REPORT
{'=' * 70}
📊 Symbol: {signal.symbol} | Style: Multi-Timeframe Analysis
⏰ Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🔗 Signal ID: {signal.signal_id}

🎯 SIGNAL ASSESSMENT:
Action: {signal.action} ({signal.strength} strength)
Confidence: {signal.confidence:.1f}% | Processing: {signal.processing_time_ms:.0f}ms

📈 CONFLUENCE ANALYSIS:
Overall Score: {signal.confluence_score:.1f}/100
Timeframe Agreement: {signal.timeframe_agreement:.1f}%
Trend Strength: {signal.trend_strength}
Volatility Risk: {signal.volatility_risk}

💰 PRICE & RISK MANAGEMENT:
Current Price: ₹{signal.current_price:.2f}
Entry Price: ₹{signal.entry_price:.2f}
Stop Loss: ₹{signal.stop_loss:.2f}
Target 1: ₹{signal.target_1:.2f}
Target 2: ₹{signal.target_2:.2f}
Target 3: ₹{signal.target_3:.2f}

⚖️ RISK METRICS:
Risk/Reward Ratio: {signal.risk_reward_ratio:.2f}
Position Size: {signal.position_size_pct:.2f}% of portfolio
Max Risk: {signal.max_risk_pct:.2f}% per trade
Risk per Share: ₹{signal.risk_per_share:.2f}

🔍 KEY LEVELS:
Support: ₹{signal.key_support:.2f}
Resistance: ₹{signal.key_resistance:.2f}

🧠 AI ANALYSIS:
{signal.ai_reasoning}

✅ BULLISH FACTORS ({len(signal.bullish_factors)}):
{self._format_report_factors(signal.bullish_factors)}

❌ BEARISH FACTORS ({len(signal.bearish_factors)}):
{self._format_report_factors(signal.bearish_factors)}

🔍 KEY INDICATORS:
{', '.join(signal.key_indicators) if signal.key_indicators else 'None specified'}

⚠️ WARNINGS:
{self._format_report_factors(signal.warning_signs)}

🤖 AI Provider: {signal.ai_provider}
        """

    def _format_report_factors(self, factors: List[str]) -> str:
        """Format factors for report"""
        if not factors:
            return "None"
        return "\n".join([f"• {factor}" for factor in factors[:5]])


# Testing function
async def test_enhanced_ai_signal_generator():
    """Test the enhanced AI signal generator"""
    print("🧪 Testing Enhanced AI Signal Generator with Confluence")
    print("=" * 70)

    generator = EnhancedAISignalGenerator()

    test_cases = [
        ("RELIANCE.NS", "swing_trade"),
        ("TCS.NS", "day_trade"),
        ("INFY.NS", "long_term")
    ]

    for symbol, style in test_cases:
        print(f"\n📊 Testing {symbol} - {style}")
        try:
            signal = await generator.generate_enhanced_signal(symbol, style)
            print(f"✅ Signal: {signal.action} | Confidence: {signal.confidence:.1f}%")
            print(f"   Confluence: {signal.confluence_score:.1f}/100 | Agreement: {signal.timeframe_agreement:.1f}%")
            print(
                f"   Entry: ₹{signal.entry_price:.2f} | Stop: ₹{signal.stop_loss:.2f} | Target: ₹{signal.target_1:.2f}")
            print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f} | Position: {signal.position_size_pct:.2f}%")

            if symbol == "RELIANCE.NS":  # Show full report for first test
                print("\n" + "=" * 70)
                print("FULL ENHANCED SIGNAL REPORT:")
                print("=" * 70)
                print(generator.format_enhanced_report(signal))
                print("=" * 70)

        except Exception as e:
            print(f"❌ Test failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Enhanced AI Signal Generator testing completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_ai_signal_generator())