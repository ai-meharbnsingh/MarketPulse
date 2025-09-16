# src/ai_trading/ai_fundamental_analyzer.py
"""
MarketPulse AI Fundamental Analyzer - Day 6
Grade A+ Fundamental Analysis with AI Document Processing
Integrates with existing technical analysis system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    """Container for fundamental analysis metrics"""
    # Valuation Ratios
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    ev_ebitda: float = 0.0
    peg_ratio: float = 0.0

    # Profitability Ratios
    roe: float = 0.0
    roa: float = 0.0
    net_margin: float = 0.0
    operating_margin: float = 0.0
    gross_margin: float = 0.0

    # Liquidity Ratios
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    cash_ratio: float = 0.0

    # Leverage Ratios
    debt_equity: float = 0.0
    interest_coverage: float = 0.0
    debt_ebitda: float = 0.0

    # Efficiency Ratios
    asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0

    # Quality Metrics
    free_cash_flow: float = 0.0
    cash_flow_quality: float = 0.0
    earnings_quality: float = 0.0

    # Growth Metrics
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    book_value_growth: float = 0.0


@dataclass
class AIFundamentalScore:
    """AI-generated fundamental analysis score"""
    overall_score: float  # 0-100
    value_score: float  # 0-100
    quality_score: float  # 0-100
    growth_score: float  # 0-100
    safety_score: float  # 0-100

    confidence_level: float  # 0-100
    investment_thesis: str
    key_risks: List[str]
    catalysts: List[str]
    fair_value_estimate: float
    recommendation: str  # BUY/HOLD/SELL

    reasoning: str
    timestamp: datetime


class AIFundamentalAnalyzer:
    """
    Professional AI-powered fundamental analysis system
    Integrates with MarketPulse technical analysis
    """

    def __init__(self, ai_engine=None, config: Dict = None):
        self.ai_engine = ai_engine or self._create_mock_ai_engine()
        self.config = config or self._default_config()

        # Integration with existing technical system
        self.technical_analyzer = None  # Will be injected

        # Fundamental analysis thresholds
        self.quality_thresholds = {
            'excellent_roe': 20.0,
            'good_roe': 15.0,
            'excellent_margin': 20.0,
            'good_margin': 10.0,
            'safe_debt_equity': 0.5,
            'max_debt_equity': 1.0,
            'min_current_ratio': 1.5,
            'excellent_current_ratio': 2.0
        }

        # Valuation thresholds (will be sector-adjusted)
        self.valuation_thresholds = {
            'cheap_pe': 15.0,
            'fair_pe': 25.0,
            'expensive_pe': 35.0,
            'cheap_pb': 2.0,
            'fair_pb': 3.0,
            'expensive_pb': 5.0
        }

    def _create_mock_ai_engine(self):
        """Mock AI engine for testing without API keys"""

        class MockAI:
            async def get_completion(self, prompt: str) -> str:
                return f"Mock AI response for: {prompt[:100]}..."

        return MockAI()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'risk_free_rate': 0.07,  # 7% Indian risk-free rate
            'market_risk_premium': 0.06,  # 6% equity risk premium
            'default_beta': 1.0,
            'min_market_cap': 1000,  # Minimum market cap in crores
            'data_lookback_years': 5
        }

    async def analyze_company_fundamentals(self,
                                           symbol: str,
                                           include_ai_analysis: bool = True) -> Tuple[
        FundamentalMetrics, Optional[AIFundamentalScore]]:
        """
        Complete fundamental analysis of a company

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            include_ai_analysis: Whether to include AI-powered analysis

        Returns:
            Tuple of (FundamentalMetrics, AIFundamentalScore)
        """
        try:
            logger.info(f"Starting fundamental analysis for {symbol}")

            # Step 1: Collect financial data
            financial_data = await self._collect_financial_data(symbol)

            # Step 2: Calculate fundamental metrics
            metrics = self._calculate_fundamental_metrics(financial_data)

            # Step 3: AI-powered analysis (optional)
            ai_score = None
            if include_ai_analysis and self.ai_engine:
                ai_score = await self._generate_ai_analysis(symbol, metrics, financial_data)

            logger.info(f"Fundamental analysis complete for {symbol}")
            return metrics, ai_score

        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {str(e)}")
            raise

    async def _collect_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Collect comprehensive financial data"""
        try:
            # Use yfinance for financial data
            ticker = yf.Ticker(symbol)

            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow

            # Get key metrics
            info = ticker.info

            # Get historical price data for valuation
            hist_data = ticker.history(period="5y")

            financial_data = {
                'info': info,
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'price_history': hist_data,
                'current_price': info.get('currentPrice', hist_data['Close'][-1] if not hist_data.empty else None)
            }

            return financial_data

        except Exception as e:
            logger.error(f"Error collecting financial data: {str(e)}")
            # Return empty structure for graceful degradation
            return {
                'info': {},
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame(),
                'price_history': pd.DataFrame(),
                'current_price': None
            }

    def _calculate_fundamental_metrics(self, financial_data: Dict[str, Any]) -> FundamentalMetrics:
        """Calculate comprehensive fundamental metrics"""
        info = financial_data.get('info', {})

        # Extract key metrics from yfinance info
        metrics = FundamentalMetrics()

        # Valuation Ratios
        metrics.pe_ratio = info.get('trailingPE', 0.0) or 0.0
        metrics.pb_ratio = info.get('priceToBook', 0.0) or 0.0
        metrics.ps_ratio = info.get('priceToSalesTrailing12Months', 0.0) or 0.0
        metrics.ev_ebitda = info.get('enterpriseToEbitda', 0.0) or 0.0
        metrics.peg_ratio = info.get('pegRatio', 0.0) or 0.0

        # Profitability Ratios
        metrics.roe = info.get('returnOnEquity', 0.0) or 0.0
        metrics.roa = info.get('returnOnAssets', 0.0) or 0.0
        metrics.net_margin = info.get('profitMargins', 0.0) or 0.0
        metrics.operating_margin = info.get('operatingMargins', 0.0) or 0.0
        metrics.gross_margin = info.get('grossMargins', 0.0) or 0.0

        # Convert percentages to proper format if needed
        if metrics.roe > 1:
            metrics.roe = metrics.roe / 100
        if metrics.roa > 1:
            metrics.roa = metrics.roa / 100

        # Liquidity Ratios
        metrics.current_ratio = info.get('currentRatio', 0.0) or 0.0
        metrics.quick_ratio = info.get('quickRatio', 0.0) or 0.0

        # Leverage Ratios
        metrics.debt_equity = info.get('debtToEquity', 0.0) or 0.0
        if metrics.debt_equity > 0:
            metrics.debt_equity = metrics.debt_equity / 100  # Convert to ratio

        # Growth Metrics
        metrics.revenue_growth = info.get('revenueGrowth', 0.0) or 0.0
        metrics.earnings_growth = info.get('earningsGrowth', 0.0) or 0.0

        # Cash Flow Metrics
        metrics.free_cash_flow = info.get('freeCashflow', 0.0) or 0.0

        return metrics

    async def _generate_ai_analysis(self,
                                    symbol: str,
                                    metrics: FundamentalMetrics,
                                    financial_data: Dict[str, Any]) -> AIFundamentalScore:
        """Generate AI-powered fundamental analysis"""

        # Prepare comprehensive prompt for AI analysis
        analysis_prompt = self._create_analysis_prompt(symbol, metrics, financial_data)

        try:
            # Get AI analysis
            ai_response = await self.ai_engine.get_completion(analysis_prompt)

            # Parse AI response and create score
            ai_score = self._parse_ai_response(ai_response, symbol, metrics)

            return ai_score

        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            # Return fallback analysis
            return self._create_fallback_analysis(symbol, metrics)

    def _create_analysis_prompt(self,
                                symbol: str,
                                metrics: FundamentalMetrics,
                                financial_data: Dict[str, Any]) -> str:
        """Create comprehensive AI analysis prompt"""

        info = financial_data.get('info', {})
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')

        prompt = f"""
        Perform comprehensive fundamental analysis for {company_name} ({symbol}):

        COMPANY OVERVIEW:
        - Sector: {sector}
        - Industry: {industry}
        - Market Cap: {info.get('marketCap', 'N/A')}
        - Current Price: {financial_data.get('current_price', 'N/A')}

        FUNDAMENTAL METRICS:
        VALUATION:
        - P/E Ratio: {metrics.pe_ratio:.2f}
        - P/B Ratio: {metrics.pb_ratio:.2f}
        - P/S Ratio: {metrics.ps_ratio:.2f}
        - EV/EBITDA: {metrics.ev_ebitda:.2f}
        - PEG Ratio: {metrics.peg_ratio:.2f}

        PROFITABILITY:
        - ROE: {metrics.roe * 100:.2f}%
        - ROA: {metrics.roa * 100:.2f}%
        - Net Margin: {metrics.net_margin * 100:.2f}%
        - Operating Margin: {metrics.operating_margin * 100:.2f}%
        - Gross Margin: {metrics.gross_margin * 100:.2f}%

        FINANCIAL HEALTH:
        - Current Ratio: {metrics.current_ratio:.2f}
        - Quick Ratio: {metrics.quick_ratio:.2f}
        - Debt/Equity: {metrics.debt_equity:.2f}

        GROWTH:
        - Revenue Growth: {metrics.revenue_growth * 100:.2f}%
        - Earnings Growth: {metrics.earnings_growth * 100:.2f}%

        ANALYSIS REQUIRED:
        1. Overall Investment Score (0-100)
        2. Component Scores:
           - Value Score (0-100): Is it undervalued?
           - Quality Score (0-100): Financial strength & management
           - Growth Score (0-100): Future growth potential
           - Safety Score (0-100): Downside protection

        3. Investment Recommendation: BUY/HOLD/SELL
        4. Fair Value Estimate (in rupees)
        5. Investment Thesis (2-3 sentences)
        6. Top 3 Key Risks
        7. Top 3 Catalysts for growth
        8. Confidence Level (0-100)

        Please provide numerical scores and clear reasoning for each component.
        Consider sector benchmarks and current market conditions in India.
        """

        return prompt

    def _parse_ai_response(self, ai_response: str, symbol: str, metrics: FundamentalMetrics) -> AIFundamentalScore:
        """Parse AI response into structured score (simplified for demo)"""

        # This is a simplified parser - in production, you'd use more sophisticated NLP
        # For now, create reasonable scores based on metrics

        # Calculate component scores based on metrics
        value_score = self._calculate_value_score(metrics)
        quality_score = self._calculate_quality_score(metrics)
        growth_score = self._calculate_growth_score(metrics)
        safety_score = self._calculate_safety_score(metrics)

        overall_score = (value_score + quality_score + growth_score + safety_score) / 4

        # Generate recommendation
        if overall_score >= 80:
            recommendation = "BUY"
        elif overall_score >= 60:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"

        # Create structured response
        ai_score = AIFundamentalScore(
            overall_score=overall_score,
            value_score=value_score,
            quality_score=quality_score,
            growth_score=growth_score,
            safety_score=safety_score,
            confidence_level=75.0,  # Default confidence
            investment_thesis=f"Based on fundamental analysis, {symbol} shows {'strong' if overall_score >= 70 else 'moderate' if overall_score >= 50 else 'weak'} fundamentals",
            key_risks=self._identify_key_risks(metrics),
            catalysts=self._identify_catalysts(metrics),
            fair_value_estimate=0.0,  # Would calculate intrinsic value
            recommendation=recommendation,
            reasoning=f"Overall score of {overall_score:.1f} based on value ({value_score:.1f}), quality ({quality_score:.1f}), growth ({growth_score:.1f}), and safety ({safety_score:.1f}) factors.",
            timestamp=datetime.now()
        )

        return ai_score

    def _calculate_value_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate value score based on valuation metrics"""
        score = 50.0  # Base score

        # P/E Ratio scoring
        if 0 < metrics.pe_ratio <= self.valuation_thresholds['cheap_pe']:
            score += 20
        elif metrics.pe_ratio <= self.valuation_thresholds['fair_pe']:
            score += 10
        elif metrics.pe_ratio >= self.valuation_thresholds['expensive_pe']:
            score -= 20

        # P/B Ratio scoring
        if 0 < metrics.pb_ratio <= self.valuation_thresholds['cheap_pb']:
            score += 15
        elif metrics.pb_ratio >= self.valuation_thresholds['expensive_pb']:
            score -= 15

        # EV/EBITDA scoring
        if 0 < metrics.ev_ebitda <= 10:
            score += 15
        elif metrics.ev_ebitda >= 20:
            score -= 10

        return max(0, min(100, score))

    def _calculate_quality_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate quality score based on profitability and financial health"""
        score = 50.0  # Base score

        # ROE scoring
        if metrics.roe >= self.quality_thresholds['excellent_roe'] / 100:
            score += 25
        elif metrics.roe >= self.quality_thresholds['good_roe'] / 100:
            score += 15
        elif metrics.roe <= 0:
            score -= 30

        # Profit margin scoring
        if metrics.net_margin >= self.quality_thresholds['excellent_margin'] / 100:
            score += 20
        elif metrics.net_margin >= self.quality_thresholds['good_margin'] / 100:
            score += 10
        elif metrics.net_margin <= 0:
            score -= 25

        # Debt levels
        if 0 <= metrics.debt_equity <= self.quality_thresholds['safe_debt_equity']:
            score += 15
        elif metrics.debt_equity >= self.quality_thresholds['max_debt_equity']:
            score -= 20

        # Liquidity
        if metrics.current_ratio >= self.quality_thresholds['excellent_current_ratio']:
            score += 10
        elif metrics.current_ratio < self.quality_thresholds['min_current_ratio']:
            score -= 15

        return max(0, min(100, score))

    def _calculate_growth_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate growth score"""
        score = 50.0  # Base score

        # Revenue growth
        if metrics.revenue_growth >= 0.20:  # 20%+ growth
            score += 25
        elif metrics.revenue_growth >= 0.10:  # 10%+ growth
            score += 15
        elif metrics.revenue_growth < 0:  # Negative growth
            score -= 20

        # Earnings growth
        if metrics.earnings_growth >= 0.20:
            score += 20
        elif metrics.earnings_growth >= 0.10:
            score += 10
        elif metrics.earnings_growth < 0:
            score -= 15

        return max(0, min(100, score))

    def _calculate_safety_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate safety score"""
        score = 50.0  # Base score

        # Financial stability
        if metrics.current_ratio >= 2.0:
            score += 20
        elif metrics.current_ratio < 1.0:
            score -= 25

        # Debt safety
        if metrics.debt_equity <= 0.3:
            score += 20
        elif metrics.debt_equity >= 1.5:
            score -= 25

        # Profitability consistency (simplified)
        if metrics.roe > 0 and metrics.net_margin > 0:
            score += 15

        return max(0, min(100, score))

    def _identify_key_risks(self, metrics: FundamentalMetrics) -> List[str]:
        """Identify key investment risks"""
        risks = []

        if metrics.debt_equity > 1.0:
            risks.append("High debt levels may indicate financial stress")

        if metrics.current_ratio < 1.5:
            risks.append("Poor liquidity position may affect operations")

        if metrics.pe_ratio > 35:
            risks.append("High valuation may limit upside potential")

        if metrics.revenue_growth < 0:
            risks.append("Declining revenues indicate business challenges")

        if not risks:
            risks.append("Standard market and sector risks apply")

        return risks[:3]  # Return top 3

    def _identify_catalysts(self, metrics: FundamentalMetrics) -> List[str]:
        """Identify growth catalysts"""
        catalysts = []

        if metrics.roe > 0.15:
            catalysts.append("Strong return on equity indicates efficient management")

        if metrics.revenue_growth > 0.15:
            catalysts.append("Strong revenue growth momentum")

        if metrics.pe_ratio < 15:
            catalysts.append("Attractive valuation provides upside potential")

        if metrics.debt_equity < 0.5:
            catalysts.append("Conservative debt levels provide financial flexibility")

        if not catalysts:
            catalysts.append("Potential sector recovery and market expansion")

        return catalysts[:3]  # Return top 3

    def _create_fallback_analysis(self, symbol: str, metrics: FundamentalMetrics) -> AIFundamentalScore:
        """Create fallback analysis when AI is unavailable"""

        # Use rule-based scoring
        value_score = self._calculate_value_score(metrics)
        quality_score = self._calculate_quality_score(metrics)
        growth_score = self._calculate_growth_score(metrics)
        safety_score = self._calculate_safety_score(metrics)

        overall_score = (value_score + quality_score + growth_score + safety_score) / 4

        if overall_score >= 75:
            recommendation = "BUY"
        elif overall_score >= 60:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"

        return AIFundamentalScore(
            overall_score=overall_score,
            value_score=value_score,
            quality_score=quality_score,
            growth_score=growth_score,
            safety_score=safety_score,
            confidence_level=60.0,  # Lower confidence for rule-based
            investment_thesis=f"Rule-based analysis suggests {recommendation} based on fundamental metrics",
            key_risks=self._identify_key_risks(metrics),
            catalysts=self._identify_catalysts(metrics),
            fair_value_estimate=0.0,
            recommendation=recommendation,
            reasoning="Analysis based on quantitative metrics and rule-based scoring",
            timestamp=datetime.now()
        )

    def integrate_with_technical_analysis(self, technical_score: Dict, fundamental_score: AIFundamentalScore) -> Dict:
        """
        Integrate fundamental analysis with existing technical analysis
        Returns combined investment decision
        """

        # Weight technical vs fundamental based on time horizon
        # Day trading: 80% technical, 20% fundamental
        # Swing trading: 60% technical, 40% fundamental
        # Long-term: 30% technical, 70% fundamental

        trading_style = "swing"  # Default - can be made configurable

        if trading_style == "day":
            tech_weight, fund_weight = 0.8, 0.2
        elif trading_style == "swing":
            tech_weight, fund_weight = 0.6, 0.4
        else:  # long-term
            tech_weight, fund_weight = 0.3, 0.7

        # Get technical score (assuming it exists)
        tech_overall = technical_score.get('overall_score', 50.0)

        # Combined score
        combined_score = (tech_overall * tech_weight) + (fundamental_score.overall_score * fund_weight)

        # Combined recommendation logic
        if combined_score >= 75:
            final_recommendation = "BUY"
        elif combined_score >= 60:
            final_recommendation = "HOLD"
        else:
            final_recommendation = "SELL"

        return {
            'combined_score': combined_score,
            'technical_score': tech_overall,
            'fundamental_score': fundamental_score.overall_score,
            'recommendation': final_recommendation,
            'confidence': min(technical_score.get('confidence', 70), fundamental_score.confidence_level),
            'technical_weight': tech_weight,
            'fundamental_weight': fund_weight,
            'trading_style': trading_style
        }


# Example usage and testing
async def test_fundamental_analyzer():
    """Test the fundamental analyzer"""
    print("🧪 Testing AI Fundamental Analyzer...")

    # Initialize analyzer
    analyzer = AIFundamentalAnalyzer()

    # Test symbols
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

    for symbol in test_symbols:
        try:
            print(f"\n📊 Analyzing {symbol}...")
            metrics, ai_score = await analyzer.analyze_company_fundamentals(symbol)

            print(f"✅ Fundamental Metrics for {symbol}:")
            print(f"   P/E Ratio: {metrics.pe_ratio:.2f}")
            print(f"   ROE: {metrics.roe * 100:.2f}%")
            print(f"   Net Margin: {metrics.net_margin * 100:.2f}%")
            print(f"   Debt/Equity: {metrics.debt_equity:.2f}")
            print(f"   Current Ratio: {metrics.current_ratio:.2f}")

            if ai_score:
                print(f"\n🤖 AI Analysis:")
                print(f"   Overall Score: {ai_score.overall_score:.1f}/100")
                print(f"   Recommendation: {ai_score.recommendation}")
                print(f"   Value Score: {ai_score.value_score:.1f}")
                print(f"   Quality Score: {ai_score.quality_score:.1f}")
                print(f"   Growth Score: {ai_score.growth_score:.1f}")
                print(f"   Safety Score: {ai_score.safety_score:.1f}")
                print(f"   Investment Thesis: {ai_score.investment_thesis}")

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")