# src/ai_trading/ai_risk_assessor.py

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from antifragile_framework.providers.api_abstraction_layer import ChatMessage


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    volatility_regime: str  # "low", "medium", "high"
    trend_direction: str  # "bullish", "bearish", "sideways"
    risk_appetite: str  # "risk_on", "risk_off", "neutral"
    uncertainty_level: float  # 0.0 to 1.0


class AIRiskAssessor:
    """
    AI-powered risk assessment for trading decisions
    Integrates market conditions with personal risk parameters
    """

    def __init__(self, ai_engine, risk_calculator):
        self.ai_engine = ai_engine
        self.risk_calculator = risk_calculator
        self.logger = logging.getLogger(__name__)

    async def assess_trade_risk(
            self,
            symbol: str,
            trade_type: str,  # "day_trade", "swing_trade", "long_term"
            entry_price: float,
            stop_loss: float,
            target_price: float,
            position_size_percent: float,
            market_data: Dict = None
    ) -> Dict:
        """
        Comprehensive AI risk assessment for a trade

        Returns:
            Complete risk analysis with recommendations
        """

        # Calculate basic risk metrics
        risk_reward_ratio = self._calculate_risk_reward(
            entry_price, stop_loss, target_price
        )

        # Get AI market condition assessment
        market_analysis = await self._ai_market_condition_analysis(symbol, market_data)

        # Get AI position sizing recommendation
        sizing_analysis = await self._ai_position_sizing_analysis(
            symbol, trade_type, position_size_percent, risk_reward_ratio
        )

        # Get AI timing assessment
        timing_analysis = await self._ai_timing_analysis(symbol, trade_type)

        # Combine all analyses
        overall_assessment = await self._ai_overall_risk_assessment(
            symbol, risk_reward_ratio, market_analysis,
            sizing_analysis, timing_analysis
        )

        return {
            "symbol": symbol,
            "trade_type": trade_type,
            "risk_reward_ratio": risk_reward_ratio,
            "market_condition": market_analysis,
            "position_sizing": sizing_analysis,
            "timing_assessment": timing_analysis,
            "overall_assessment": overall_assessment,
            "timestamp": datetime.now().isoformat(),
            "ai_recommendation": self._extract_recommendation(overall_assessment)
        }

    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk-reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0

    async def _ai_market_condition_analysis(self, symbol: str, market_data: Dict) -> Dict:
        """AI analysis of current market conditions"""

        market_prompt = f"""
        Analyze current market conditions for {symbol}:

        Recent market data: {market_data or "Use general market knowledge"}

        Assess:
        1. Overall market volatility regime (low/medium/high)
        2. Trend direction (bullish/bearish/sideways)
        3. Risk appetite in market (risk_on/risk_off/neutral)
        4. Key risks and opportunities
        5. Optimal trading timeframe for current conditions

        Provide specific, actionable insights for risk management.
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "openai": ["gpt-4o"],
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[ChatMessage(role="user", content=market_prompt)],
                max_estimated_cost_usd=0.015,
                request_id=f"market_analysis_{symbol}"
            )

            return {
                "status": "success",
                "analysis": response.content,
                "model_used": response.model_used
            }

        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _ai_position_sizing_analysis(
            self, symbol: str, trade_type: str,
            proposed_size: float, risk_reward: float
    ) -> Dict:
        """AI analysis of position sizing appropriateness"""

        sizing_prompt = f"""
        Evaluate this position sizing decision:

        Symbol: {symbol}
        Trade type: {trade_type}
        Proposed position size: {proposed_size}% of capital
        Risk-reward ratio: {risk_reward:.2f}

        Consider:
        1. Is position size appropriate for trade type?
        2. Does it match risk-reward ratio quality?
        3. Account for current market volatility
        4. Risk of correlation with existing positions

        Recommend: increase/decrease/maintain position size with reasoning.
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "openai": ["gpt-4o"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[ChatMessage(role="user", content=sizing_prompt)],
                max_estimated_cost_usd=0.01,
                request_id=f"sizing_analysis_{symbol}"
            )

            return {
                "status": "success",
                "analysis": response.content,
                "model_used": response.model_used
            }

        except Exception as e:
            self.logger.error(f"Position sizing analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _ai_timing_analysis(self, symbol: str, trade_type: str) -> Dict:
        """AI analysis of trade timing"""

        timing_prompt = f"""
        Assess timing for this {trade_type} in {symbol}:

        Consider:
        1. Current market phase (accumulation/markup/distribution/decline)
        2. Seasonal factors affecting {symbol}
        3. Upcoming events (earnings, news, economic data)
        4. Technical timing signals
        5. Optimal entry/exit timeframes

        Recommend: immediate/wait/avoid with specific timing guidance.
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "openai": ["gpt-4o"],
                    "google_gemini": ["gemini-1.5-flash-latest"],
                    "anthropic": ["claude-3-5-sonnet-20240620"]
                },
                messages=[ChatMessage(role="user", content=timing_prompt)],
                max_estimated_cost_usd=0.01,
                request_id=f"timing_analysis_{symbol}"
            )

            return {
                "status": "success",
                "analysis": response.content,
                "model_used": response.model_used
            }

        except Exception as e:
            self.logger.error(f"Timing analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _ai_overall_risk_assessment(
            self, symbol: str, risk_reward: float,
            market_analysis: Dict, sizing_analysis: Dict, timing_analysis: Dict
    ) -> Dict:
        """Final AI overall risk assessment"""

        overall_prompt = f"""
        Provide final risk assessment for {symbol} trade:

        Risk-Reward Ratio: {risk_reward:.2f}

        Market Analysis: {market_analysis.get('analysis', 'N/A')}
        Position Sizing: {sizing_analysis.get('analysis', 'N/A')}
        Timing Analysis: {timing_analysis.get('analysis', 'N/A')}

        Provide:
        1. Overall risk score (1-10, where 10 is highest risk)
        2. Key risk factors
        3. Final recommendation: EXECUTE/MODIFY/WAIT/AVOID
        4. Specific modifications if needed
        5. One-sentence summary for quick decision making
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "openai": ["gpt-4o"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[ChatMessage(role="user", content=overall_prompt)],
                max_estimated_cost_usd=0.015,
                request_id=f"overall_risk_{symbol}"
            )

            return {
                "status": "success",
                "analysis": response.content,
                "model_used": response.model_used
            }

        except Exception as e:
            self.logger.error(f"Overall risk assessment failed: {e}")
            return {"status": "error", "message": str(e)}

    def _extract_recommendation(self, overall_assessment: Dict) -> str:
        """Extract clear recommendation from AI analysis"""
        if overall_assessment.get("status") == "error":
            return "MANUAL_REVIEW_REQUIRED"

        analysis = overall_assessment.get("analysis", "").upper()

        if "EXECUTE" in analysis:
            return "EXECUTE"
        elif "MODIFY" in analysis:
            return "MODIFY"
        elif "WAIT" in analysis:
            return "WAIT"
        elif "AVOID" in analysis:
            return "AVOID"
        else:
            return "UNCLEAR_REVIEW_REQUIRED"