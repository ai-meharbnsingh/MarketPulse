# src/ai_trading/ai_signal_generator_fixed.py

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
sys.path.append(os.path.join(current_dir, '../..'))

# Add the framework path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../01_Framework_Core'))

try:
    from antifragile_framework.core.ai_engine import AIEngine
    from antifragile_framework.core.config import FrameworkConfig

    FRAMEWORK_AVAILABLE = True
except ImportError:
    print("⚠️ Antifragile Framework not found, using mock AI engine for testing")
    FRAMEWORK_AVAILABLE = False

# Import our modules with error handling
try:
    from professional_technical_analyzer import ProfessionalTechnicalAnalyzer, TechnicalSignal
    from multi_timeframe_collector import MultiTimeframeCollector
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


@dataclass
class AITradingSignal:
    """Enhanced trading signal with AI analysis"""
    # Required fields (no defaults) come first
    symbol: str
    timeframe: str
    timestamp: datetime
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    strength: str  # WEAK, MODERATE, STRONG
    entry_price: float
    stop_loss: float
    target_1: float
    risk_reward_ratio: float
    position_size_pct: float  # % of portfolio
    max_risk_pct: float  # % risk per trade
    ai_reasoning: str
    technical_confluence: int  # Number of supporting indicators
    market_regime: str  # BULL, BEAR, SIDEWAYS
    volatility_regime: str  # LOW, MEDIUM, HIGH
    key_indicators: List[str]
    warning_signs: List[str]
    catalysts: List[str]
    ai_provider: str
    processing_time_ms: float
    signal_id: str

    # Optional fields (with defaults) come last
    target_2: Optional[float] = None
    target_3: Optional[float] = None


class MockAIEngine:
    """Mock AI engine for testing when framework is not available"""

    def __init__(self):
        self.provider = "MockAI"

    async def get_completion(self, prompt: str, **kwargs) -> str:
        """Generate mock AI response based on prompt analysis"""
        await asyncio.sleep(0.1)  # Simulate API call delay

        prompt_upper = prompt.upper()

        # Simple analysis based on prompt content
        if "BUY" in prompt_upper and "SIGNAL" in prompt_upper:
            confidence = 75 if "STRONG" in prompt_upper else 65
            return f"""
            {{
                "action": "BUY",
                "confidence": {confidence},
                "reasoning": "Technical confluence suggests bullish momentum with multiple indicators aligning. RSI showing oversold conditions with volume confirmation supporting upward move.",
                "entry_range": "2450-2470",
                "stop_loss": "2400",
                "targets": ["2520", "2580", "2650"],
                "risk_reward": 2.8,
                "market_regime": "BULL",
                "volatility_regime": "MEDIUM",
                "key_indicators": ["RSI_Oversold", "Volume_Breakout", "EMA_Crossover"],
                "key_catalysts": ["Technical breakout", "Volume confirmation", "Sector momentum"],
                "warnings": ["Market volatility", "Overall market conditions"],
                "time_horizon": "3-7 days",
                "conviction_level": "MEDIUM"
            }}
            """
        elif "SELL" in prompt_upper and "SIGNAL" in prompt_upper:
            confidence = 70 if "STRONG" in prompt_upper else 60
            return f"""
            {{
                "action": "SELL",
                "confidence": {confidence},
                "reasoning": "Technical indicators showing overbought conditions with bearish divergence. Volume declining on recent moves up suggesting weakness.",
                "entry_range": "2400-2420",
                "stop_loss": "2470",
                "targets": ["2350", "2300", "2250"],
                "risk_reward": 2.2,
                "market_regime": "BEAR",
                "volatility_regime": "MEDIUM",
                "key_indicators": ["RSI_Overbought", "MACD_Bearish", "Volume_Decline"],
                "key_catalysts": ["Technical breakdown", "Profit taking", "Sector weakness"],
                "warnings": ["Support levels nearby", "Potential bounce"],
                "time_horizon": "2-5 days",
                "conviction_level": "MEDIUM"
            }}
            """
        else:
            return f"""
            {{
                "action": "HOLD",
                "confidence": 45,
                "reasoning": "Mixed technical signals with no clear directional bias. Market showing consolidation pattern with equal buy and sell pressure.",
                "entry_range": "Current levels",
                "stop_loss": "N/A",
                "targets": ["N/A"],
                "risk_reward": 0,
                "market_regime": "SIDEWAYS",
                "volatility_regime": "LOW",
                "key_indicators": ["Mixed_Signals"],
                "key_catalysts": ["Awaiting catalyst"],
                "warnings": ["Choppy market conditions", "Low conviction"],
                "time_horizon": "Wait for clarity",
                "conviction_level": "LOW"
            }}
            """


class AISignalGenerator:
    """
    Advanced AI-powered trading signal generator
    Integrates with Antifragile Framework for multi-AI analysis
    """

    def __init__(self):
        self.technical_analyzer = ProfessionalTechnicalAnalyzer()
        self.timeframe_collector = MultiTimeframeCollector()

        # Initialize AI engine
        self.ai_engine = None
        self._initialize_ai_engine()

        # Signal generation templates
        self.signal_templates = {
            'day_trade': {
                'timeframes': ['5m', '15m', '1h'],
                'risk_per_trade': 1.0,  # 1% per trade
                'holding_period': 'intraday',
                'stop_loss_atr': 1.5
            },
            'swing_trade': {
                'timeframes': ['1h', '4h', '1d'],
                'risk_per_trade': 2.0,  # 2% per trade
                'holding_period': '3-10 days',
                'stop_loss_atr': 2.0
            },
            'long_term': {
                'timeframes': ['1d', '1w'],
                'risk_per_trade': 3.0,  # 3% per trade
                'holding_period': '1-6 months',
                'stop_loss_atr': 3.0
            }
        }

    def _initialize_ai_engine(self):
        """Initialize AI engine with fallback to mock"""
        try:
            if FRAMEWORK_AVAILABLE:
                config = FrameworkConfig()
                self.ai_engine = AIEngine(config)
                print("✅ Connected to Antifragile AI Framework")
            else:
                raise ImportError("Framework not available")
        except Exception as e:
            print(f"⚠️ Using mock AI engine: {e}")
            self.ai_engine = MockAIEngine()

    async def generate_ai_signal(self,
                                 symbol: str,
                                 trading_style: str = 'swing_trade',
                                 custom_analysis: Optional[str] = None) -> AITradingSignal:
        """
        Generate comprehensive AI-powered trading signal
        """
        start_time = datetime.now()
        signal_id = f"{symbol}_{trading_style}_{int(start_time.timestamp())}"

        print(f"🤖 Generating AI signal for {symbol} ({trading_style})")

        try:
            # Step 1: Collect multi-timeframe data
            timeframes = self.signal_templates[trading_style]['timeframes']
            print(f"📊 Analyzing timeframes: {timeframes}")

            # Get comprehensive technical data
            data_collection = await self.timeframe_collector.collect_symbol_data_parallel(symbol)
            if not data_collection:
                raise Exception("No market data available")

            # Step 2: Professional technical analysis
            primary_tf_data = list(data_collection.values())[0]
            enhanced_data = self.technical_analyzer.add_all_indicators(primary_tf_data.copy())
            technical_summary = self.technical_analyzer.get_technical_summary(enhanced_data)

            # Step 3: Multi-timeframe signals
            tf_signals = self.timeframe_collector.get_latest_signals(data_collection)

            # Step 4: Generate AI analysis prompt
            ai_prompt = self._create_comprehensive_prompt(
                symbol=symbol,
                trading_style=trading_style,
                technical_summary=technical_summary,
                timeframe_signals=tf_signals,
                market_data=enhanced_data.tail(5).to_dict('records'),  # Last 5 bars
                custom_analysis=custom_analysis
            )

            # Step 5: Get AI analysis
            ai_response = await self.ai_engine.get_completion(ai_prompt)
            ai_analysis = self._parse_ai_response(ai_response)

            # Step 6: Calculate risk management parameters
            current_price = enhanced_data['Close'].iloc[-1]
            atr = enhanced_data.get('ATR', pd.Series([current_price * 0.02])).iloc[-1]
            if pd.isna(atr):
                atr = current_price * 0.02  # 2% default ATR

            risk_params = self._calculate_risk_parameters(
                current_price=current_price,
                atr=atr,
                ai_analysis=ai_analysis,
                trading_style=trading_style
            )

            # Step 7: Determine market regimes
            market_regime = self._determine_market_regime(enhanced_data)
            volatility_regime = self._determine_volatility_regime(enhanced_data)

            # Step 8: Create comprehensive AI signal
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            signal = AITradingSignal(
                symbol=symbol,
                timeframe=f"Multi-TF ({'/'.join(timeframes)})",
                timestamp=start_time,
                action=ai_analysis.get('action', 'HOLD'),
                confidence=ai_analysis.get('confidence', 50),
                strength=self._determine_signal_strength(ai_analysis.get('confidence', 50)),
                entry_price=risk_params['entry_price'],
                stop_loss=risk_params['stop_loss'],
                target_1=risk_params['target_1'],
                target_2=risk_params.get('target_2'),
                target_3=risk_params.get('target_3'),
                risk_reward_ratio=risk_params['risk_reward_ratio'],
                position_size_pct=risk_params['position_size_pct'],
                max_risk_pct=self.signal_templates[trading_style]['risk_per_trade'],
                ai_reasoning=ai_analysis.get('reasoning', 'No reasoning provided'),
                technical_confluence=len(
                    [s for s in technical_summary['signals'] if s['signal'] == ai_analysis.get('action', 'HOLD')]),
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                key_indicators=ai_analysis.get('key_indicators', []),
                warning_signs=ai_analysis.get('warnings', []),
                catalysts=ai_analysis.get('key_catalysts', []),
                ai_provider=getattr(self.ai_engine, 'provider', 'mock'),
                processing_time_ms=processing_time,
                signal_id=signal_id
            )

            print(f"✅ Generated {signal.action} signal with {signal.confidence:.1f}% confidence")
            return signal

        except Exception as e:
            print(f"❌ Error generating AI signal: {e}")
            # Return safe HOLD signal
            return self._create_safe_signal(symbol, trading_style, str(e), signal_id)

    def _create_comprehensive_prompt(self,
                                     symbol: str,
                                     trading_style: str,
                                     technical_summary: Dict,
                                     timeframe_signals: Dict,
                                     market_data: List[Dict],
                                     custom_analysis: Optional[str] = None) -> str:
        """Create comprehensive AI analysis prompt"""

        prompt = f"""
You are an expert quantitative trader and technical analyst. Analyze {symbol} and provide a comprehensive trading signal.

**TRADING CONTEXT:**
- Symbol: {symbol}
- Trading Style: {trading_style}
- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Risk per Trade: {self.signal_templates[trading_style]['risk_per_trade']}%
- Holding Period: {self.signal_templates[trading_style]['holding_period']}

**TECHNICAL ANALYSIS SUMMARY:**
- Current Price: ₹{technical_summary.get('price', 0):.2f}
- Signal Bias: {technical_summary.get('signal_bias', 'NEUTRAL')}
- Total Signals: {technical_summary.get('total_signals', 0)} (Buy: {technical_summary.get('buy_signals', 0)}, Sell: {technical_summary.get('sell_signals', 0)})
- Trend Strength: {technical_summary.get('technical_scores', {}).get('trend_strength', 50):.1f}/100
- Momentum: {technical_summary.get('technical_scores', {}).get('momentum_composite', 50):.1f}/100
- Volume Profile: {technical_summary.get('technical_scores', {}).get('volume_profile', 50):.1f}/100
- Volatility Index: {technical_summary.get('technical_scores', {}).get('volatility_index', 50):.1f}/100

**KEY TECHNICAL LEVELS:**
- Resistance: ₹{technical_summary.get('key_levels', {}).get('resistance', 0):.2f}
- Support: ₹{technical_summary.get('key_levels', {}).get('support', 0):.2f}
- Pivot: ₹{technical_summary.get('key_levels', {}).get('pivot', 0):.2f}

**ACTIVE TECHNICAL SIGNALS:**
{self._format_signals_for_prompt(technical_summary.get('signals', []))}

**MULTI-TIMEFRAME ANALYSIS:**
{self._format_timeframe_signals_for_prompt(timeframe_signals)}

**RECENT PRICE ACTION:**
{self._format_market_data_for_prompt(market_data)}

{f"**CUSTOM ANALYSIS REQUEST:** {custom_analysis}" if custom_analysis else ""}

**REQUIRED OUTPUT FORMAT (JSON):**
Provide your analysis in this exact JSON format:

{{
    "action": "BUY|SELL|HOLD",
    "confidence": 0-100,
    "reasoning": "Detailed explanation of your decision with specific technical references",
    "entry_range": "price range for entry (e.g., '2450-2470')",
    "stop_loss": "stop loss level",
    "targets": ["target1", "target2", "target3"],
    "risk_reward": calculated_ratio,
    "market_regime": "BULL|BEAR|SIDEWAYS",
    "volatility_regime": "LOW|MEDIUM|HIGH",
    "key_indicators": ["list", "of", "supporting", "indicators"],
    "key_catalysts": ["list", "of", "key", "catalysts"],
    "warnings": ["list", "of", "potential", "risks"],
    "time_horizon": "expected holding period",
    "conviction_level": "HIGH|MEDIUM|LOW"
}}

**ANALYSIS GUIDELINES:**
1. Consider the multi-timeframe confluence
2. Weight recent signals more heavily
3. Factor in current market volatility
4. Provide specific entry/exit levels
5. Include risk management parameters
6. Be conservative with confidence if signals are mixed
7. Consider the trading style timeframe
8. Mention specific indicator values in reasoning

Analyze thoroughly and provide your professional trading recommendation.
        """

        return prompt

    def _format_signals_for_prompt(self, signals: List[Dict]) -> str:
        """Format technical signals for AI prompt"""
        if not signals:
            return "No active signals"

        formatted = []
        for signal in signals:
            formatted.append(
                f"- {signal['indicator']}: {signal['signal']} (strength: {signal['strength']:.2f}) - {signal['reasoning']}")

        return "\n".join(formatted)

    def _format_timeframe_signals_for_prompt(self, tf_signals: Dict) -> str:
        """Format timeframe signals for AI prompt"""
        if not tf_signals:
            return "No timeframe data available"

        formatted = []
        for tf, signal in tf_signals.items():
            formatted.append(
                f"- {tf}: Trend={signal['trend']}, RSI={signal['rsi']:.1f}, Volume={signal['volume']}, Weight={signal['weight']:.2f}")

        return "\n".join(formatted)

    def _format_market_data_for_prompt(self, market_data: List[Dict]) -> str:
        """Format recent market data for AI prompt"""
        if not market_data:
            return "No recent data available"

        formatted = []
        for i, bar in enumerate(market_data[-3:]):  # Last 3 bars
            formatted.append(
                f"Bar {i + 1}: O={bar.get('Open', 0):.2f}, H={bar.get('High', 0):.2f}, L={bar.get('Low', 0):.2f}, C={bar.get('Close', 0):.2f}, V={bar.get('Volume', 0):,.0f}")

        return "\n".join(formatted)

    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]

            # Find JSON content
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return json.loads(response)

        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse AI response as JSON: {e}")
            # Fallback parsing
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict:
        """Fallback parsing for non-JSON responses"""
        # Simple keyword-based parsing
        response_upper = response.upper()

        if 'BUY' in response_upper and 'SELL' not in response_upper:
            action = 'BUY'
            confidence = 70
        elif 'SELL' in response_upper and 'BUY' not in response_upper:
            action = 'SELL'
            confidence = 70
        else:
            action = 'HOLD'
            confidence = 50

        return {
            'action': action,
            'confidence': confidence,
            'reasoning': response[:500],  # First 500 chars
            'market_regime': 'SIDEWAYS',
            'key_indicators': [],
            'key_catalysts': [],
            'warnings': ['AI response parsing failed']
        }

    def _calculate_risk_parameters(self,
                                   current_price: float,
                                   atr: float,
                                   ai_analysis: Dict,
                                   trading_style: str) -> Dict:
        """Calculate comprehensive risk management parameters"""

        action = ai_analysis.get('action', 'HOLD')
        confidence = ai_analysis.get('confidence', 50)

        if action == 'HOLD':
            return {
                'entry_price': current_price,
                'stop_loss': current_price,
                'target_1': current_price,
                'risk_reward_ratio': 0,
                'position_size_pct': 0
            }

        # Parse AI targets or calculate based on ATR
        try:
            targets = ai_analysis.get('targets', [])
            if isinstance(targets, list) and targets:
                target_1 = float(targets[0]) if targets[0] != "N/A" else current_price * (
                    1.02 if action == 'BUY' else 0.98)
            else:
                # Default ATR-based targets
                atr_multiplier = self.signal_templates[trading_style]['stop_loss_atr']
                target_1 = current_price + (atr * atr_multiplier * (1 if action == 'BUY' else -1))
        except:
            target_1 = current_price * (1.02 if action == 'BUY' else 0.98)

        # Parse or calculate stop loss
        try:
            stop_loss_str = ai_analysis.get('stop_loss', '')
            stop_loss = float(stop_loss_str) if stop_loss_str and stop_loss_str != "N/A" else None
        except:
            stop_loss = None

        if not stop_loss:
            # ATR-based stop loss
            atr_multiplier = self.signal_templates[trading_style]['stop_loss_atr']
            stop_loss = current_price - (atr * atr_multiplier * (1 if action == 'BUY' else -1))

        # Calculate position size based on confidence and risk
        base_risk = self.signal_templates[trading_style]['risk_per_trade']
        confidence_multiplier = confidence / 100.0
        position_size_pct = base_risk * confidence_multiplier * 0.5  # Conservative sizing

        # Calculate risk-reward ratio
        risk_per_share = abs(current_price - stop_loss)
        reward_per_share = abs(target_1 - current_price)
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        return {
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_1 * (1.01 if action == 'BUY' else 0.99) if len(
                ai_analysis.get('targets', [])) > 1 else None,
            'target_3': target_1 * (1.02 if action == 'BUY' else 0.98) if len(
                ai_analysis.get('targets', [])) > 2 else None,
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_size_pct': round(position_size_pct, 2)
        }

    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime"""
        if df.empty or len(df) < 50:
            return "SIDEWAYS"

        # Use trend strength and moving averages
        trend_strength = df.get('TREND_STRENGTH', pd.Series([50])).iloc[-1]

        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            current_price = df['Close'].iloc[-1]

            if pd.notna(sma_50) and pd.notna(sma_200) and current_price > sma_50 > sma_200 and trend_strength > 60:
                return "BULL"
            elif pd.notna(sma_50) and pd.notna(sma_200) and current_price < sma_50 < sma_200 and trend_strength < 40:
                return "BEAR"

        return "SIDEWAYS"

    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine current volatility regime"""
        if 'VOLATILITY_INDEX' not in df.columns:
            return "MEDIUM"

        vol_index = df['VOLATILITY_INDEX'].iloc[-1]

        if vol_index > 70:
            return "HIGH"
        elif vol_index < 30:
            return "LOW"
        else:
            return "MEDIUM"

    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength category"""
        if confidence >= 80:
            return "STRONG"
        elif confidence >= 60:
            return "MODERATE"
        else:
            return "WEAK"

    def _create_safe_signal(self, symbol: str, trading_style: str, error_msg: str, signal_id: str) -> AITradingSignal:
        """Create safe HOLD signal when error occurs"""
        return AITradingSignal(
            symbol=symbol,
            timeframe="ERROR",
            timestamp=datetime.now(),
            action="HOLD",
            confidence=0,
            strength="WEAK",
            entry_price=0,
            stop_loss=0,
            target_1=0,
            risk_reward_ratio=0,
            position_size_pct=0,
            max_risk_pct=0,
            ai_reasoning=f"Error in analysis: {error_msg}",
            technical_confluence=0,
            market_regime="UNKNOWN",
            volatility_regime="UNKNOWN",
            key_indicators=[],
            warning_signs=[error_msg],
            catalysts=[],
            ai_provider="error",
            processing_time_ms=0,
            signal_id=signal_id
        )

    def format_signal_report(self, signal: AITradingSignal) -> str:
        """Format signal into readable report"""

        report = f"""
🤖 AI TRADING SIGNAL REPORT
{'=' * 50}
📊 Symbol: {signal.symbol}
⏰ Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🎯 Action: {signal.action} ({signal.strength})
📈 Confidence: {signal.confidence:.1f}%
⚡ Processing Time: {signal.processing_time_ms:.0f}ms

💰 PRICE TARGETS:
Entry: ₹{signal.entry_price:.2f}
Stop Loss: ₹{signal.stop_loss:.2f}
Target 1: ₹{signal.target_1:.2f}
{f'Target 2: ₹{signal.target_2:.2f}' if signal.target_2 else ''}
{f'Target 3: ₹{signal.target_3:.2f}' if signal.target_3 else ''}

⚖️ RISK MANAGEMENT:
Risk/Reward: {signal.risk_reward_ratio:.2f}
Position Size: {signal.position_size_pct:.2f}% of portfolio
Max Risk: {signal.max_risk_pct:.2f}% per trade

🧠 AI ANALYSIS:
{signal.ai_reasoning}

📊 MARKET CONTEXT:
Market Regime: {signal.market_regime}
Volatility: {signal.volatility_regime}
Technical Confluence: {signal.technical_confluence} indicators

🔍 KEY INDICATORS:
{', '.join(signal.key_indicators) if signal.key_indicators else 'None specified'}

🚀 CATALYSTS:
{', '.join(signal.catalysts) if signal.catalysts else 'None specified'}

⚠️ WARNINGS:
{', '.join(signal.warning_signs) if signal.warning_signs else 'None'}

🔗 Signal ID: {signal.signal_id}
🤖 AI Provider: {signal.ai_provider}
        """

        return report.strip()


# Testing function
async def test_ai_signal_generator():
    """Test the AI signal generator"""
    print("🧪 Testing AI Signal Generator")
    print("=" * 60)

    generator = AISignalGenerator()

    # Test different trading styles
    test_cases = [
        ("RELIANCE.NS", "swing_trade"),
        ("TCS.NS", "day_trade"),
        ("INFY.NS", "long_term")
    ]

    for symbol, style in test_cases:
        print(f"\n📊 Testing {symbol} - {style}")
        try:
            signal = await generator.generate_ai_signal(symbol, style)
            print(f"✅ Generated {signal.action} signal with {signal.confidence:.1f}% confidence")
            print(f"   Entry: ₹{signal.entry_price:.2f}, Target: ₹{signal.target_1:.2f}, Stop: ₹{signal.stop_loss:.2f}")
            print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")

            # Print full report for first test
            if symbol == "RELIANCE.NS":
                print("\n" + "=" * 60)
                print("FULL SIGNAL REPORT:")
                print("=" * 60)
                print(generator.format_signal_report(signal))
                print("=" * 60)

        except Exception as e:
            print(f"❌ Test failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ AI Signal Generator testing completed!")


if __name__ == "__main__":
    asyncio.run(test_ai_signal_generator())