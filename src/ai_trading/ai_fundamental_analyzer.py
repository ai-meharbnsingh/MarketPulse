# src/ai_trading/ai_fundamental_analyzer.py - FIXED VERSION

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import asyncio
import traceback


@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics"""
    symbol: str
    # Valuation metrics
    pe_ratio: float
    pb_ratio: float
    peg_ratio: float
    price_to_sales: float
    ev_ebitda: float

    # Profitability metrics
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    gross_margin: float
    operating_margin: float
    net_margin: float

    # Financial health
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    interest_coverage: float

    # Growth metrics
    revenue_growth: float
    earnings_growth: float

    # Market data
    market_cap: float
    enterprise_value: float
    dividend_yield: float

    # Calculated scores
    value_score: float  # 0-100
    quality_score: float  # 0-100
    growth_score: float  # 0-100
    overall_score: float  # 0-100

    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence_level: float  # 0-1

    timestamp: datetime


class AIFundamentalAnalyzer:
    """FIXED: AI-enhanced fundamental analysis system"""

    def __init__(self):  # REMOVED ai_framework parameter
        """Initialize the fundamental analyzer"""
        self.risk_free_rate = 0.06  # 6% for Indian context
        self.market_return = 0.12  # 12% expected market return

        # Scoring weights for different aspects
        self.scoring_weights = {
            'value': 0.30,
            'quality': 0.40,
            'growth': 0.30
        }

        # Benchmark ratios for Indian market
        self.benchmarks = {
            'pe_ratio': {'excellent': 15, 'good': 20, 'fair': 25, 'poor': 30},
            'pb_ratio': {'excellent': 1.5, 'good': 2.5, 'fair': 3.5, 'poor': 5.0},
            'roe': {'excellent': 20, 'good': 15, 'fair': 10, 'poor': 5},
            'debt_equity': {'excellent': 0.3, 'good': 0.5, 'fair': 1.0, 'poor': 2.0},
            'current_ratio': {'excellent': 2.0, 'good': 1.5, 'fair': 1.0, 'poor': 0.8}
        }

    async def analyze_company(self, symbol: str) -> FundamentalMetrics:
        """Comprehensive fundamental analysis of a company"""
        print(f"📊 Starting fundamental analysis for {symbol}")

        try:
            # Get company data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow

            # Calculate metrics
            metrics = self._calculate_all_metrics(symbol, info, financials, balance_sheet, cash_flow)

            # Calculate component scores
            value_score = self._calculate_value_score(metrics)
            quality_score = self._calculate_quality_score(metrics)
            growth_score = self._calculate_growth_score(metrics)

            # Overall score and recommendation
            overall_score = (
                    value_score * self.scoring_weights['value'] +
                    quality_score * self.scoring_weights['quality'] +
                    growth_score * self.scoring_weights['growth']
            )

            recommendation, confidence = self._generate_recommendation(overall_score, metrics)

            # Update metrics with scores
            metrics.value_score = value_score
            metrics.quality_score = quality_score
            metrics.growth_score = growth_score
            metrics.overall_score = overall_score
            metrics.recommendation = recommendation
            metrics.confidence_level = confidence
            metrics.timestamp = datetime.now()

            print(f"✅ Analysis completed for {symbol}: {recommendation} ({overall_score:.1f}/100)")
            return metrics

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return self._create_default_metrics(symbol)

    def _calculate_all_metrics(self, symbol: str, info: dict, financials: pd.DataFrame,
                               balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> FundamentalMetrics:
        """Calculate all fundamental metrics"""

        try:
            # Market data
            market_cap = info.get('marketCap', 0)
            enterprise_value = info.get('enterpriseValue', market_cap)
            current_price = info.get('currentPrice', info.get('previousClose', 0))

            # Valuation ratios
            pe_ratio = info.get('trailingPE', info.get('forwardPE', 0))
            pb_ratio = info.get('priceToBook', 0)
            peg_ratio = info.get('pegRatio', 0)
            price_to_sales = info.get('priceToSalesTrailing12Months', 0)
            ev_ebitda = info.get('enterpriseToEbitda', 0)

            # Profitability ratios
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            roa = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
            gross_margin = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
            operating_margin = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
            net_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0

            # Financial health
            debt_to_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)

            # Interest coverage (calculated from financials if available)
            interest_coverage = 0
            if not financials.empty:
                try:
                    ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 0
                    interest_expense = abs(
                        financials.loc['Interest Expense'].iloc[0]) if 'Interest Expense' in financials.index else 1
                    interest_coverage = ebit / interest_expense if interest_expense > 0 else 0
                except:
                    interest_coverage = 0

            # Growth metrics
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0

            # Other metrics
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0

            return FundamentalMetrics(
                symbol=symbol,
                pe_ratio=self._safe_float(pe_ratio),
                pb_ratio=self._safe_float(pb_ratio),
                peg_ratio=self._safe_float(peg_ratio),
                price_to_sales=self._safe_float(price_to_sales),
                ev_ebitda=self._safe_float(ev_ebitda),
                roe=self._safe_float(roe),
                roa=self._safe_float(roa),
                gross_margin=self._safe_float(gross_margin),
                operating_margin=self._safe_float(operating_margin),
                net_margin=self._safe_float(net_margin),
                debt_to_equity=self._safe_float(debt_to_equity),
                current_ratio=self._safe_float(current_ratio),
                quick_ratio=self._safe_float(quick_ratio),
                interest_coverage=self._safe_float(interest_coverage),
                revenue_growth=self._safe_float(revenue_growth),
                earnings_growth=self._safe_float(earnings_growth),
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                dividend_yield=self._safe_float(dividend_yield),
                value_score=0.0,  # Will be calculated
                quality_score=0.0,  # Will be calculated
                growth_score=0.0,  # Will be calculated
                overall_score=0.0,  # Will be calculated
                recommendation="",  # Will be calculated
                confidence_level=0.0,  # Will be calculated
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return self._create_default_metrics(symbol)

    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        if value is None or pd.isna(value):
            return 0.0
        try:
            return float(value)
        except:
            return 0.0

    def _calculate_value_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate value score (0-100)"""
        score = 0
        factors = 0

        # P/E ratio scoring
        if metrics.pe_ratio > 0:
            if metrics.pe_ratio <= self.benchmarks['pe_ratio']['excellent']:
                score += 25
            elif metrics.pe_ratio <= self.benchmarks['pe_ratio']['good']:
                score += 20
            elif metrics.pe_ratio <= self.benchmarks['pe_ratio']['fair']:
                score += 15
            elif metrics.pe_ratio <= self.benchmarks['pe_ratio']['poor']:
                score += 10
            else:
                score += 5
            factors += 1

        # P/B ratio scoring
        if metrics.pb_ratio > 0:
            if metrics.pb_ratio <= self.benchmarks['pb_ratio']['excellent']:
                score += 25
            elif metrics.pb_ratio <= self.benchmarks['pb_ratio']['good']:
                score += 20
            elif metrics.pb_ratio <= self.benchmarks['pb_ratio']['fair']:
                score += 15
            elif metrics.pb_ratio <= self.benchmarks['pb_ratio']['poor']:
                score += 10
            else:
                score += 5
            factors += 1

        # PEG ratio scoring
        if metrics.peg_ratio > 0:
            if metrics.peg_ratio <= 1.0:
                score += 25
            elif metrics.peg_ratio <= 1.5:
                score += 20
            elif metrics.peg_ratio <= 2.0:
                score += 15
            else:
                score += 10
            factors += 1

        # Price to Sales scoring
        if metrics.price_to_sales > 0:
            if metrics.price_to_sales <= 1.0:
                score += 25
            elif metrics.price_to_sales <= 2.0:
                score += 20
            elif metrics.price_to_sales <= 3.0:
                score += 15
            else:
                score += 10
            factors += 1

        return (score / max(factors, 1)) if factors > 0 else 50.0

    def _calculate_quality_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate quality score (0-100)"""
        score = 0
        factors = 0

        # ROE scoring
        if metrics.roe > 0:
            if metrics.roe >= self.benchmarks['roe']['excellent']:
                score += 30
            elif metrics.roe >= self.benchmarks['roe']['good']:
                score += 25
            elif metrics.roe >= self.benchmarks['roe']['fair']:
                score += 20
            else:
                score += 15
            factors += 1

        # ROA scoring
        if metrics.roa > 0:
            if metrics.roa >= 10:
                score += 20
            elif metrics.roa >= 5:
                score += 15
            else:
                score += 10
            factors += 1

        # Debt to Equity scoring (lower is better)
        if metrics.debt_to_equity >= 0:
            if metrics.debt_to_equity <= self.benchmarks['debt_equity']['excellent']:
                score += 25
            elif metrics.debt_to_equity <= self.benchmarks['debt_equity']['good']:
                score += 20
            elif metrics.debt_to_equity <= self.benchmarks['debt_equity']['fair']:
                score += 15
            else:
                score += 10
            factors += 1

        # Current ratio scoring
        if metrics.current_ratio > 0:
            if metrics.current_ratio >= self.benchmarks['current_ratio']['excellent']:
                score += 25
            elif metrics.current_ratio >= self.benchmarks['current_ratio']['good']:
                score += 20
            elif metrics.current_ratio >= self.benchmarks['current_ratio']['fair']:
                score += 15
            else:
                score += 10
            factors += 1

        return (score / max(factors, 1)) if factors > 0 else 50.0

    def _calculate_growth_score(self, metrics: FundamentalMetrics) -> float:
        """Calculate growth score (0-100)"""
        score = 0
        factors = 0

        # Revenue growth scoring
        if metrics.revenue_growth != 0:
            if metrics.revenue_growth >= 20:
                score += 50
            elif metrics.revenue_growth >= 15:
                score += 40
            elif metrics.revenue_growth >= 10:
                score += 30
            elif metrics.revenue_growth >= 5:
                score += 20
            else:
                score += 10
            factors += 1

        # Earnings growth scoring
        if metrics.earnings_growth != 0:
            if metrics.earnings_growth >= 25:
                score += 50
            elif metrics.earnings_growth >= 20:
                score += 40
            elif metrics.earnings_growth >= 15:
                score += 30
            elif metrics.earnings_growth >= 10:
                score += 20
            else:
                score += 10
            factors += 1

        return (score / max(factors, 1)) if factors > 0 else 50.0

    def _generate_recommendation(self, overall_score: float, metrics: FundamentalMetrics) -> Tuple[str, float]:
        """Generate investment recommendation and confidence level"""

        # Base recommendation on overall score
        if overall_score >= 80:
            recommendation = "STRONG_BUY"
            confidence = 0.9
        elif overall_score >= 65:
            recommendation = "BUY"
            confidence = 0.8
        elif overall_score >= 45:
            recommendation = "HOLD"
            confidence = 0.7
        elif overall_score >= 30:
            recommendation = "SELL"
            confidence = 0.8
        else:
            recommendation = "STRONG_SELL"
            confidence = 0.9

        # Adjust confidence based on data quality
        data_quality_factors = sum([
            1 if metrics.pe_ratio > 0 else 0,
            1 if metrics.pb_ratio > 0 else 0,
            1 if metrics.roe > 0 else 0,
            1 if metrics.revenue_growth != 0 else 0,
            1 if metrics.debt_to_equity >= 0 else 0
        ])

        confidence_adjustment = min(data_quality_factors / 5.0, 1.0)
        confidence = confidence * confidence_adjustment

        return recommendation, confidence

    def _create_default_metrics(self, symbol: str) -> FundamentalMetrics:
        """Create default metrics when data is unavailable"""
        return FundamentalMetrics(
            symbol=symbol,
            pe_ratio=0.0, pb_ratio=0.0, peg_ratio=0.0, price_to_sales=0.0, ev_ebitda=0.0,
            roe=0.0, roa=0.0, gross_margin=0.0, operating_margin=0.0, net_margin=0.0,
            debt_to_equity=0.0, current_ratio=0.0, quick_ratio=0.0, interest_coverage=0.0,
            revenue_growth=0.0, earnings_growth=0.0,
            market_cap=0, enterprise_value=0, dividend_yield=0.0,
            value_score=50.0, quality_score=50.0, growth_score=50.0, overall_score=50.0,
            recommendation="HOLD", confidence_level=0.3,
            timestamp=datetime.now()
        )

    def format_analysis_report(self, metrics: FundamentalMetrics) -> str:
        """Format fundamental analysis into readable report"""

        report = f"""
📊 FUNDAMENTAL ANALYSIS REPORT
{'=' * 50}
Symbol: {metrics.symbol}
Analysis Date: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

🎯 INVESTMENT RECOMMENDATION: {metrics.recommendation}
Confidence Level: {metrics.confidence_level:.1%}
Overall Score: {metrics.overall_score:.1f}/100

📈 COMPONENT SCORES:
Value Score: {metrics.value_score:.1f}/100
Quality Score: {metrics.quality_score:.1f}/100  
Growth Score: {metrics.growth_score:.1f}/100

💰 VALUATION METRICS:
P/E Ratio: {metrics.pe_ratio:.2f}
P/B Ratio: {metrics.pb_ratio:.2f}
PEG Ratio: {metrics.peg_ratio:.2f}
Price/Sales: {metrics.price_to_sales:.2f}
EV/EBITDA: {metrics.ev_ebitda:.2f}

💪 PROFITABILITY METRICS:
ROE: {metrics.roe:.1f}%
ROA: {metrics.roa:.1f}%
Gross Margin: {metrics.gross_margin:.1f}%
Operating Margin: {metrics.operating_margin:.1f}%
Net Margin: {metrics.net_margin:.1f}%

🏦 FINANCIAL HEALTH:
Debt/Equity: {metrics.debt_to_equity:.2f}
Current Ratio: {metrics.current_ratio:.2f}
Quick Ratio: {metrics.quick_ratio:.2f}
Interest Coverage: {metrics.interest_coverage:.2f}

📈 GROWTH METRICS:
Revenue Growth: {metrics.revenue_growth:.1f}%
Earnings Growth: {metrics.earnings_growth:.1f}%

💹 MARKET DATA:
Market Cap: ₹{metrics.market_cap:,.0f}
Enterprise Value: ₹{metrics.enterprise_value:,.0f}
Dividend Yield: {metrics.dividend_yield:.1f}%
        """

        return report.strip()


# Testing function
async def test_fundamental_analyzer():
    """Test the fundamental analyzer"""
    print("🧪 Testing AI Fundamental Analyzer")
    print("=" * 50)

    analyzer = AIFundamentalAnalyzer()

    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    for symbol in test_symbols:
        try:
            print(f"\n📊 Analyzing {symbol}...")
            metrics = await analyzer.analyze_company(symbol)

            print(f"✅ Analysis complete for {symbol}")
            print(f"Recommendation: {metrics.recommendation}")
            print(f"Overall Score: {metrics.overall_score:.1f}/100")
            print(f"Confidence: {metrics.confidence_level:.1%}")

            # Print detailed report
            print(analyzer.format_analysis_report(metrics))
            print("\n" + "=" * 50)

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(test_fundamental_analyzer())