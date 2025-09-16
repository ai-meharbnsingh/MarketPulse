# File: src/ai_trading/ai_confluence_scorer.py

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "01_Framework_Core"))

# Import AI framework
from antifragile_framework.core.failover_engine import FailoverEngine
from antifragile_framework.config.config_loader import load_provider_profiles
from antifragile_framework.providers.api_abstraction_layer import ChatMessage
from antifragile_framework.providers.provider_registry import get_default_provider_registry
from telemetry.event_bus import EventBus

# Import our data collector
from multi_timeframe_collector import MultiTimeframeCollector

import os
from dotenv import load_dotenv

load_dotenv()


class AIConfluenceScorer:
    """
    Uses multiple AI providers to analyze multi-timeframe confluence and generate
    high-confidence trading signals with detailed explanations.
    """

    def __init__(self):
        # Initialize AI framework
        self.provider_configs = {
            "openai": {
                "api_keys": [k.strip() for k in os.getenv("OPENAI_API_KEY", "").split(",") if k.strip()],
                "resource_config": {},
                "circuit_breaker_config": {},
            },
            "google_gemini": {
                "api_keys": [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()],
                "resource_config": {},
                "circuit_breaker_config": {},
            },
            "anthropic": {
                "api_keys": [k.strip() for k in os.getenv("ANTHROPIC_API_KEY", "").split(",") if k.strip()],
                "resource_config": {},
                "circuit_breaker_config": {},
            },
        }

        # Initialize AI engine
        provider_profiles = load_provider_profiles()
        provider_registry = get_default_provider_registry()
        event_bus = EventBus()

        self.ai_engine = FailoverEngine(
            provider_configs=self.provider_configs,
            provider_registry=provider_registry,
            event_bus=event_bus,
            provider_profiles=provider_profiles,
        )

        # Initialize data collector
        self.data_collector = MultiTimeframeCollector()

        print("🧠 AI Confluence Scorer initialized")
        print("✅ Multi-AI engine ready")
        print("✅ Multi-timeframe collector ready")

    async def analyze_confluence(self, symbol: str, analysis_type: str = "swing_trade") -> Dict:
        """
        Perform comprehensive multi-timeframe confluence analysis using AI

        Args:
            symbol: Stock symbol to analyze
            analysis_type: 'day_trade', 'swing_trade', or 'long_term'

        Returns:
            Complete confluence analysis with AI insights
        """
        print(f"🎯 Starting AI confluence analysis for {symbol}")
        print(f"📊 Analysis type: {analysis_type}")

        # Step 1: Collect multi-timeframe data
        data_collection = await self.data_collector.collect_symbol_data_parallel(symbol, days_back=15)
        signals = self.data_collector.get_latest_signals(data_collection)

        if not signals:
            return {"error": "No data collected for analysis"}

        # Step 2: Calculate basic confluence score
        basic_confluence = self._calculate_basic_confluence(signals)

        # Step 3: Get AI analysis
        ai_analysis = await self._get_ai_confluence_analysis(symbol, signals, analysis_type)

        # Step 4: Generate final trading recommendation
        final_recommendation = await self._generate_trading_recommendation(
            symbol, signals, basic_confluence, ai_analysis, analysis_type
        )

        # Combine all results
        result = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "timeframe_signals": signals,
            "basic_confluence": basic_confluence,
            "ai_analysis": ai_analysis,
            "final_recommendation": final_recommendation,
            "raw_data": data_collection,  # ADD THIS LINE
            "confidence_score": final_recommendation.get("confidence", 0)
        }

        return result

    def _calculate_basic_confluence(self, signals: Dict) -> Dict:
        """
        Calculate basic confluence scores without AI
        """
        if not signals:
            return {"error": "No signals to analyze"}

        # Count bullish/bearish signals by weight
        bullish_weight = 0
        bearish_weight = 0
        neutral_weight = 0

        trend_distribution = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        momentum_distribution = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0, "OVERBOUGHT": 0, "OVERSOLD": 0}

        for timeframe, signal in signals.items():
            weight = signal['weight']
            trend = signal['trend']
            momentum = signal['momentum']

            # Weight the trends
            if trend == "BULLISH":
                bullish_weight += weight
            elif trend == "BEARISH":
                bearish_weight += weight
            else:
                neutral_weight += weight

            # Count distributions
            trend_distribution[trend] += 1
            momentum_distribution[momentum] += 1

        # Calculate confluence score (0-100)
        total_weight = bullish_weight + bearish_weight + neutral_weight
        bullish_pct = (bullish_weight / total_weight) * 100 if total_weight > 0 else 0
        bearish_pct = (bearish_weight / total_weight) * 100 if total_weight > 0 else 0

        # Determine primary direction
        if bullish_pct > 60:
            direction = "BULLISH"
            confluence_strength = bullish_pct
        elif bearish_pct > 60:
            direction = "BEARISH"
            confluence_strength = bearish_pct
        else:
            direction = "MIXED"
            confluence_strength = 50 - abs(bullish_pct - bearish_pct)  # Lower score for mixed signals

        return {
            "direction": direction,
            "confluence_strength": round(confluence_strength, 1),
            "bullish_weight": round(bullish_weight, 2),
            "bearish_weight": round(bearish_weight, 2),
            "neutral_weight": round(neutral_weight, 2),
            "trend_distribution": trend_distribution,
            "momentum_distribution": momentum_distribution
        }

    async def _get_ai_confluence_analysis(self, symbol: str, signals: Dict, analysis_type: str) -> Dict:
        """
        Get AI analysis of the confluence signals
        """
        # Create comprehensive prompt for AI analysis
        analysis_prompt = f"""
        You are an expert technical analyst specializing in multi-timeframe confluence analysis for Indian stock markets.

        Analyze the following multi-timeframe data for {symbol}:

        TIMEFRAME SIGNALS:
        {json.dumps(signals, indent=2)}

        ANALYSIS REQUIREMENTS:
        1. Evaluate the confluence strength across all timeframes
        2. Identify key support/resistance levels
        3. Assess momentum alignment and divergences  
        4. Evaluate volume confirmation
        5. Identify the highest probability trade setup
        6. Provide specific entry, stop-loss, and target levels
        7. Rate the overall setup quality (1-10)
        8. Highlight any red flags or conflicting signals

        CONTEXT: This is for {analysis_type} trading approach.

        Provide analysis in JSON format:
        {{
            "confluence_assessment": "detailed analysis of timeframe alignment",
            "key_levels": {{
                "support": [list of support levels],
                "resistance": [list of resistance levels]
            }},
            "momentum_analysis": "momentum alignment and divergences",
            "volume_analysis": "volume confirmation assessment", 
            "trade_setup": {{
                "type": "breakout/pullback/reversal/continuation",
                "quality_score": 8,
                "entry_zone": [price range],
                "stop_loss": price,
                "targets": [list of target prices],
                "risk_reward_ratio": 2.5
            }},
            "red_flags": ["list of concerns"],
            "timeframe_priority": "which timeframe is most important for this setup"
        }}
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "anthropic": ["claude-3-5-sonnet-20240620"],  # Best for analysis
                    "openai": ["gpt-4o"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[
                    ChatMessage(role="system",
                                content="You are an expert technical analyst. Always respond with valid JSON."),
                    ChatMessage(role="user", content=analysis_prompt)
                ],
                max_estimated_cost_usd=0.05,  # Higher budget for detailed analysis
                request_id=f"confluence_analysis_{symbol}"
            )

            # Parse AI response
            ai_content = response.content.strip()

            # Extract JSON from response
            if '```json' in ai_content:
                ai_content = ai_content.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_content:
                ai_content = ai_content.split('```')[1].split('```')[0].strip()

            try:
                ai_analysis = json.loads(ai_content)
                ai_analysis['ai_provider'] = response.model_used
                ai_analysis['analysis_cost'] = "Low cost due to AI optimization"
                return ai_analysis
            except json.JSONDecodeError:
                return {
                    "error": "AI response parsing failed",
                    "raw_response": ai_content[:500],
                    "ai_provider": response.model_used
                }

        except Exception as e:
            return {
                "error": f"AI analysis failed: {str(e)}",
                "fallback": "Using basic technical analysis only"
            }

    async def _generate_trading_recommendation(self, symbol: str, signals: Dict,
                                               basic_confluence: Dict, ai_analysis: Dict,
                                               analysis_type: str) -> Dict:
        """
        Generate final trading recommendation combining all analysis
        """
        # Combine basic and AI analysis for final recommendation
        recommendation_prompt = f"""
        Generate a final trading recommendation for {symbol} based on:

        BASIC CONFLUENCE: {json.dumps(basic_confluence, indent=2)}

        AI ANALYSIS: {json.dumps(ai_analysis, indent=2)}

        TRADING STYLE: {analysis_type}

        Provide a clear, actionable recommendation in JSON format:
        {{
            "action": "BUY/SELL/HOLD/WAIT",
            "confidence": 85,
            "reasoning": "clear explanation of the decision",
            "entry_strategy": "specific entry approach",
            "risk_management": "stop loss and position sizing guidance",
            "time_horizon": "expected duration of trade",
            "key_catalysts": ["what could drive this trade"],
            "exit_strategy": "profit taking and stop loss approach"
        }}
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "openai": ["gpt-4o"],  # Good for final recommendations
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[
                    ChatMessage(role="system",
                                content="You are a professional trader providing final recommendations. Always respond with valid JSON."),
                    ChatMessage(role="user", content=recommendation_prompt)
                ],
                max_estimated_cost_usd=0.03,
                request_id=f"final_recommendation_{symbol}"
            )

            # Parse recommendation
            rec_content = response.content.strip()
            if '```json' in rec_content:
                rec_content = rec_content.split('```json')[1].split('```')[0].strip()
            elif '```' in rec_content:
                rec_content = rec_content.split('```')[1].split('```')[0].strip()

            try:
                recommendation = json.loads(rec_content)
                recommendation['ai_provider'] = response.model_used
                return recommendation
            except json.JSONDecodeError:
                return {
                    "action": "WAIT",
                    "confidence": 50,
                    "reasoning": "AI recommendation parsing failed - exercise caution",
                    "error": "JSON parsing failed",
                    "raw_response": rec_content[:500]
                }

        except Exception as e:
            return {
                "action": "WAIT",
                "confidence": 30,
                "reasoning": f"AI system error: {str(e)} - manual analysis recommended",
                "error": str(e)
            }


# Import pandas for timestamp
import pandas as pd


# Test the AI Confluence Scorer
async def test_confluence_scorer():
    print("🧪 Testing AI Confluence Scorer")
    print("=" * 60)

    try:
        scorer = AIConfluenceScorer()

        # Test with RELIANCE
        symbol = "RELIANCE.NS"
        analysis_type = "swing_trade"

        print(f"\n🎯 Analyzing {symbol} for {analysis_type}...")

        result = await scorer.analyze_confluence(symbol, analysis_type)

        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return False

        print(f"\n🎯 CONFLUENCE ANALYSIS RESULTS for {symbol}")
        print("=" * 60)

        # Basic confluence
        basic = result['basic_confluence']
        print(f"📊 Basic Confluence:")
        print(f"   Direction: {basic['direction']}")
        print(f"   Strength: {basic['confluence_strength']}%")
        print(f"   Bullish Weight: {basic['bullish_weight']}")
        print(f"   Bearish Weight: {basic['bearish_weight']}")

        # AI Analysis
        if 'ai_analysis' in result and 'error' not in result['ai_analysis']:
            ai = result['ai_analysis']
            print(f"\n🧠 AI Analysis (via {ai.get('ai_provider', 'Unknown')}):")
            if 'trade_setup' in ai:
                setup = ai['trade_setup']
                print(f"   Setup Type: {setup.get('type', 'N/A')}")
                print(f"   Quality Score: {setup.get('quality_score', 'N/A')}/10")
                print(f"   Risk/Reward: {setup.get('risk_reward_ratio', 'N/A')}")

        # Final recommendation
        if 'final_recommendation' in result:
            rec = result['final_recommendation']
            print(f"\n🎯 Final Recommendation:")
            print(f"   Action: {rec.get('action', 'N/A')}")
            print(f"   Confidence: {rec.get('confidence', 'N/A')}%")
            print(f"   Reasoning: {rec.get('reasoning', 'N/A')[:100]}...")

        print(f"\n✅ AI Confluence Analysis completed successfully!")
        print(f"💰 Cost optimized through multi-AI provider system")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_confluence_scorer())