# src/ai_trading/complete_fundamental_system.py - FIXED VERSION

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import yfinance as yf


@dataclass
class ComprehensiveFundamentalAnalysis:
    """Complete fundamental analysis result"""
    symbol: str
    company_name: str

    # Core metrics
    fundamental_score: float  # 0-100
    news_sentiment: float  # -1 to +1
    document_quality: float  # 0-1
    overall_rating: str  # STRONG_BUY to STRONG_SELL

    # Component analyses
    financial_metrics: Dict
    sentiment_analysis: Dict
    document_insights: Dict

    # Investment thesis
    investment_thesis: str
    key_risks: List[str]
    key_opportunities: List[str]

    # Recommendations
    target_price: float
    stop_loss: float
    time_horizon: str
    position_size_recommendation: float

    timestamp: datetime


class CompleteFundamentalSystem:
    """FIXED: Complete integrated fundamental analysis system"""

    def __init__(self):  # REMOVED ai_framework parameter
        """Initialize the complete fundamental system"""
        # Initialize components - using direct imports to avoid circular dependencies
        print("🏗️ Initializing Complete Fundamental Analysis System")

        # Component weights for final scoring
        self.component_weights = {
            'fundamental': 0.50,  # 50% weight to financial metrics
            'sentiment': 0.30,  # 30% weight to news sentiment
            'document': 0.20  # 20% weight to document quality
        }

        # Investment rating thresholds
        self.rating_thresholds = {
            'STRONG_BUY': 80,
            'BUY': 65,
            'HOLD': 45,
            'SELL': 30,
            'STRONG_SELL': 0
        }

        print("✅ Complete Fundamental System initialized")

    async def analyze_complete_fundamental(self, symbol: str) -> ComprehensiveFundamentalAnalysis:
        """Perform complete fundamental analysis using all components"""
        print(f"🔍 Starting comprehensive fundamental analysis for {symbol}")

        try:
            # Get company info for context
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)

            # Component 1: Fundamental metrics analysis
            fundamental_metrics = await self._analyze_fundamental_metrics(symbol)

            # Component 2: News sentiment analysis
            sentiment_analysis = await self._analyze_news_sentiment(symbol)

            # Component 3: Document insights
            document_insights = await self._analyze_document_quality(symbol)

            # Calculate weighted overall score
            overall_score = (
                    fundamental_metrics['score'] * self.component_weights['fundamental'] +
                    sentiment_analysis['sentiment_score'] * 50 + 50 * self.component_weights[
                        'sentiment'] +  # Normalize -1,1 to 0,100
                    document_insights['quality_score'] * 100 * self.component_weights['document']
            )

            # Generate investment rating
            rating = self._determine_investment_rating(overall_score)

            # Generate investment thesis and recommendations
            thesis, risks, opportunities = self._generate_investment_thesis(
                symbol, fundamental_metrics, sentiment_analysis, document_insights
            )

            # Calculate price targets and position sizing
            target_price, stop_loss = self._calculate_price_targets(symbol, fundamental_metrics)
            position_size = self._recommend_position_size(overall_score, risks)

            return ComprehensiveFundamentalAnalysis(
                symbol=symbol,
                company_name=company_name,
                fundamental_score=round(overall_score, 1),
                news_sentiment=sentiment_analysis['sentiment_score'],
                document_quality=document_insights['quality_score'],
                overall_rating=rating,
                financial_metrics=fundamental_metrics,
                sentiment_analysis=sentiment_analysis,
                document_insights=document_insights,
                investment_thesis=thesis,
                key_risks=risks,
                key_opportunities=opportunities,
                target_price=target_price,
                stop_loss=stop_loss,
                time_horizon=self._determine_time_horizon(overall_score),
                position_size_recommendation=position_size,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Error in complete fundamental analysis for {symbol}: {e}")
            return self._create_default_analysis(symbol)

    async def _analyze_fundamental_metrics(self, symbol: str) -> Dict:
        """Analyze fundamental metrics using simplified approach"""
        try:
            print(f"📊 Analyzing fundamental metrics for {symbol}")

            # Get basic financial data
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Key metrics
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            debt_equity = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0

            # Simple scoring system
            score = 50  # Start neutral

            # PE ratio scoring (lower is better for value)
            if pe_ratio > 0:
                if pe_ratio <= 15:
                    score += 15
                elif pe_ratio <= 25:
                    score += 10
                elif pe_ratio <= 35:
                    score += 5
                else:
                    score -= 5

            # ROE scoring (higher is better)
            if roe > 0:
                if roe >= 20:
                    score += 15
                elif roe >= 15:
                    score += 10
                elif roe >= 10:
                    score += 5

            # Revenue growth scoring
            if revenue_growth > 0:
                if revenue_growth >= 20:
                    score += 10
                elif revenue_growth >= 10:
                    score += 5

            # Debt control scoring (lower debt is better)
            if debt_equity >= 0:
                if debt_equity <= 0.3:
                    score += 10
                elif debt_equity <= 0.5:
                    score += 5
                elif debt_equity >= 2.0:
                    score -= 10

            score = max(0, min(100, score))  # Clamp to 0-100

            return {
                'score': score,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'roe': roe,
                'debt_equity': debt_equity,
                'revenue_growth': revenue_growth,
                'market_cap': info.get('marketCap', 0),
                'analysis_quality': 'basic_metrics'
            }

        except Exception as e:
            print(f"⚠️ Error in fundamental metrics analysis: {e}")
            return {
                'score': 50,
                'pe_ratio': 0,
                'pb_ratio': 0,
                'roe': 0,
                'debt_equity': 0,
                'revenue_growth': 0,
                'market_cap': 0,
                'analysis_quality': 'error_fallback'
            }

    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment using simplified approach"""
        try:
            print(f"📰 Analyzing news sentiment for {symbol}")

            # Placeholder for news sentiment analysis
            # In production, this would integrate with news APIs

            # Simulated sentiment analysis
            import random
            random.seed(hash(symbol))  # Consistent results per symbol

            sentiment_score = random.uniform(-0.3, 0.3)  # Conservative sentiment range

            sentiment_label = "POSITIVE" if sentiment_score > 0.1 else "NEGATIVE" if sentiment_score < -0.1 else "NEUTRAL"

            news_count = random.randint(3, 15)

            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'news_articles_analyzed': news_count,
                'confidence': 0.7,
                'key_themes': ['Earnings', 'Market Position', 'Growth'],
                'analysis_method': 'simulated_placeholder'
            }

        except Exception as e:
            print(f"⚠️ Error in news sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'news_articles_analyzed': 0,
                'confidence': 0.3,
                'key_themes': ['Analysis pending'],
                'analysis_method': 'error_fallback'
            }

    async def _analyze_document_quality(self, symbol: str) -> Dict:
        """Analyze document quality using simplified approach"""
        try:
            print(f"📄 Analyzing document quality for {symbol}")

            # Placeholder for document analysis
            # In production, this would analyze annual reports, earnings calls, etc.

            # Simulated document quality analysis
            import random
            random.seed(hash(symbol + "docs"))

            quality_score = random.uniform(0.4, 0.8)  # Reasonable quality range

            return {
                'quality_score': quality_score,
                'documents_analyzed': random.randint(2, 8),
                'key_insights': [
                    'Management commentary analysis',
                    'Financial disclosure quality',
                    'Forward guidance assessment'
                ],
                'transparency_score': quality_score * 0.9,
                'analysis_method': 'simulated_placeholder'
            }

        except Exception as e:
            print(f"⚠️ Error in document quality analysis: {e}")
            return {
                'quality_score': 0.5,
                'documents_analyzed': 0,
                'key_insights': ['Analysis pending'],
                'transparency_score': 0.5,
                'analysis_method': 'error_fallback'
            }

    def _determine_investment_rating(self, overall_score: float) -> str:
        """Determine investment rating from overall score"""
        if overall_score >= self.rating_thresholds['STRONG_BUY']:
            return "STRONG_BUY"
        elif overall_score >= self.rating_thresholds['BUY']:
            return "BUY"
        elif overall_score >= self.rating_thresholds['HOLD']:
            return "HOLD"
        elif overall_score >= self.rating_thresholds['SELL']:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _generate_investment_thesis(self, symbol: str, fundamentals: Dict,
                                    sentiment: Dict, documents: Dict) -> Tuple[str, List[str], List[str]]:
        """Generate investment thesis, risks, and opportunities"""

        # Investment thesis based on analysis
        score = fundamentals['score']
        sentiment_label = sentiment['sentiment_label']

        if score >= 70:
            thesis = f"Strong fundamental position with {sentiment_label.lower()} market sentiment. " \
                     f"Company shows solid financial metrics and growth potential."
        elif score >= 50:
            thesis = f"Moderate fundamental strength with {sentiment_label.lower()} sentiment. " \
                     f"Mixed signals suggest cautious optimism."
        else:
            thesis = f"Weak fundamental position with {sentiment_label.lower()} sentiment. " \
                     f"Significant challenges evident in financial metrics."

        # Generate risks
        risks = []
        if fundamentals.get('pe_ratio', 0) > 30:
            risks.append("High valuation multiples")
        if fundamentals.get('debt_equity', 0) > 1.0:
            risks.append("Elevated debt levels")
        if fundamentals.get('revenue_growth', 0) < 5:
            risks.append("Slow revenue growth")
        if sentiment['sentiment_score'] < -0.1:
            risks.append("Negative market sentiment")

        if not risks:
            risks = ["General market risk", "Sector-specific challenges"]

        # Generate opportunities
        opportunities = []
        if fundamentals.get('revenue_growth', 0) > 15:
            opportunities.append("Strong revenue growth momentum")
        if fundamentals.get('roe', 0) > 18:
            opportunities.append("Excellent return on equity")
        if sentiment['sentiment_score'] > 0.1:
            opportunities.append("Positive market sentiment")
        if fundamentals.get('pe_ratio', 0) < 20 and fundamentals.get('pe_ratio', 0) > 0:
            opportunities.append("Attractive valuation")

        if not opportunities:
            opportunities = ["Market recovery potential", "Operational improvements"]

        return thesis, risks[:3], opportunities[:3]

    def _calculate_price_targets(self, symbol: str, fundamentals: Dict) -> Tuple[float, float]:
        """Calculate target price and stop loss"""
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]

            # Simple target calculation based on fundamental score
            score = fundamentals['score']

            if score >= 70:
                upside = 0.15  # 15% upside for strong fundamentals
                downside = 0.08  # 8% stop loss
            elif score >= 50:
                upside = 0.08  # 8% upside for moderate fundamentals
                downside = 0.10  # 10% stop loss
            else:
                upside = 0.03  # 3% upside for weak fundamentals
                downside = 0.12  # 12% stop loss

            target_price = current_price * (1 + upside)
            stop_loss = current_price * (1 - downside)

            return round(target_price, 2), round(stop_loss, 2)

        except Exception as e:
            print(f"⚠️ Error calculating price targets: {e}")
            return 0.0, 0.0

    def _recommend_position_size(self, overall_score: float, risks: List[str]) -> float:
        """Recommend position size as percentage of portfolio"""

        # Base position size on overall score and risk count
        base_size = 0.05  # 5% base position

        if overall_score >= 80:
            size_multiplier = 2.0  # Up to 10%
        elif overall_score >= 65:
            size_multiplier = 1.5  # Up to 7.5%
        elif overall_score >= 45:
            size_multiplier = 1.0  # 5%
        else:
            size_multiplier = 0.6  # 3%

        # Reduce size based on number of risks
        risk_penalty = len(risks) * 0.1

        final_size = base_size * size_multiplier * (1 - risk_penalty)

        # Clamp between 1% and 15%
        return max(0.01, min(0.15, final_size))

    def _determine_time_horizon(self, overall_score: float) -> str:
        """Determine recommended time horizon"""
        if overall_score >= 70:
            return "Long-term (12+ months)"
        elif overall_score >= 50:
            return "Medium-term (6-12 months)"
        else:
            return "Short-term (3-6 months)"

    def _create_default_analysis(self, symbol: str) -> ComprehensiveFundamentalAnalysis:
        """Create default analysis when processing fails"""
        return ComprehensiveFundamentalAnalysis(
            symbol=symbol,
            company_name=symbol,
            fundamental_score=50.0,
            news_sentiment=0.0,
            document_quality=0.5,
            overall_rating="HOLD",
            financial_metrics={'score': 50, 'analysis_quality': 'default'},
            sentiment_analysis={'sentiment_score': 0, 'sentiment_label': 'NEUTRAL'},
            document_insights={'quality_score': 0.5, 'key_insights': ['Analysis pending']},
            investment_thesis="Analysis pending - insufficient data",
            key_risks=["Data availability risk"],
            key_opportunities=["Analysis completion opportunity"],
            target_price=0.0,
            stop_loss=0.0,
            time_horizon="Medium-term (6-12 months)",
            position_size_recommendation=0.03,
            timestamp=datetime.now()
        )

    def format_complete_analysis_report(self, analysis: ComprehensiveFundamentalAnalysis) -> str:
        """Format complete analysis into comprehensive report"""

        report = f"""
🏢 COMPLETE FUNDAMENTAL ANALYSIS
{'=' * 60}
Company: {analysis.company_name} ({analysis.symbol})
Analysis Date: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

🎯 OVERALL ASSESSMENT:
Investment Rating: {analysis.overall_rating}
Fundamental Score: {analysis.fundamental_score:.1f}/100
News Sentiment: {analysis.news_sentiment:.2f}
Document Quality: {analysis.document_quality:.1%}

📈 PRICE TARGETS & RECOMMENDATIONS:
Current Target Price: ₹{analysis.target_price:.2f}
Stop Loss Level: ₹{analysis.stop_loss:.2f}
Time Horizon: {analysis.time_horizon}
Position Size: {analysis.position_size_recommendation:.1%} of portfolio

💭 INVESTMENT THESIS:
{analysis.investment_thesis}

⚠️ KEY RISKS ({len(analysis.key_risks)}):
{chr(10).join(['• ' + risk for risk in analysis.key_risks])}

🚀 KEY OPPORTUNITIES ({len(analysis.key_opportunities)}):
{chr(10).join(['• ' + opportunity for opportunity in analysis.key_opportunities])}

📊 COMPONENT BREAKDOWN:
Financial Metrics Score: {analysis.financial_metrics['score']:.1f}/100
Sentiment Analysis: {analysis.sentiment_analysis['sentiment_label']}
Document Quality: {analysis.document_insights['quality_score']:.1%}

⏰ Analysis completed with {self.component_weights['fundamental']:.0%} fundamental weight, {self.component_weights['sentiment']:.0%} sentiment weight, {self.component_weights['document']:.0%} document weight
        """

        return report.strip()


# Testing function
async def test_complete_system():
    """Test the complete fundamental system"""
    print("🧪 Testing Complete Fundamental Analysis System")
    print("=" * 60)

    system = CompleteFundamentalSystem()

    test_symbols = ["RELIANCE.NS", "TCS.NS"]

    for symbol in test_symbols:
        try:
            print(f"\n🏢 Analyzing {symbol} with complete system...")
            analysis = await system.analyze_complete_fundamental(symbol)

            print(f"✅ Complete analysis finished for {symbol}")
            print(f"Rating: {analysis.overall_rating}")
            print(f"Fundamental Score: {analysis.fundamental_score:.1f}/100")
            print(f"Target Price: ₹{analysis.target_price:.2f}")
            print(f"Position Size: {analysis.position_size_recommendation:.1%}")

            # Print full report
            print("\n" + "=" * 60)
            print("COMPREHENSIVE ANALYSIS REPORT:")
            print("=" * 60)
            print(system.format_complete_analysis_report(analysis))
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(test_complete_system())