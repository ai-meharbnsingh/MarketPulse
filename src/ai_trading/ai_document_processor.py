# src/ai_trading/ai_document_processor.py
"""
MarketPulse AI Document Processor - Day 6
Processes earnings calls, annual reports, news articles for fundamental analysis
Integrates with AI Fundamental Analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentAnalysis:
    """Container for document analysis results"""
    document_type: str  # 'earnings_call', 'annual_report', 'news', 'filing'
    sentiment_score: float  # -1 to +1
    key_themes: List[str]
    financial_highlights: List[str]
    risk_factors: List[str]
    management_outlook: str
    confidence_level: float
    processing_timestamp: datetime
    source_url: str = ""


@dataclass
class CompanyNewsAnalysis:
    """Analysis of company news and documents"""
    symbol: str
    overall_sentiment: float  # -1 to +1
    news_volume: int
    key_developments: List[str]
    analyst_sentiment: str  # 'POSITIVE', 'NEUTRAL', 'NEGATIVE'
    document_analyses: List[DocumentAnalysis]
    summary: str
    last_updated: datetime


class AIDocumentProcessor:
    """
    AI-powered document processing for fundamental analysis
    Processes earnings calls, reports, news for investment insights
    """

    def __init__(self, ai_engine=None, config: Dict = None):
        self.ai_engine = ai_engine or self._create_mock_ai_engine()
        self.config = config or self._default_config()

        # News sources and APIs
        self.news_sources = {
            'economic_times': 'https://economictimes.indiatimes.com',
            'business_standard': 'https://www.business-standard.com',
            'money_control': 'https://www.moneycontrol.com',
            'livemint': 'https://www.livemint.com'
        }

        # Document processing parameters
        self.sentiment_keywords = {
            'positive': ['growth', 'expansion', 'profitable', 'strong', 'excellent', 'optimistic', 'bullish',
                         'increase', 'boost'],
            'negative': ['decline', 'loss', 'weak', 'challenging', 'difficult', 'bearish', 'decrease', 'drop',
                         'concern']
        }

    def _create_mock_ai_engine(self):
        """Mock AI engine for testing"""

        class MockAI:
            async def get_completion(self, prompt: str) -> str:
                return f"Mock analysis: Document processed successfully. Sentiment appears positive based on key themes."

        return MockAI()

    def _default_config(self) -> Dict:
        return {
            'max_news_articles': 20,
            'news_lookback_days': 30,
            'min_article_length': 200,
            'sentiment_threshold': 0.1
        }

    async def process_company_documents(self, symbol: str) -> CompanyNewsAnalysis:
        """
        Process all available documents for a company

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')

        Returns:
            CompanyNewsAnalysis with comprehensive document insights
        """
        try:
            logger.info(f"Processing documents for {symbol}")

            # Clean symbol for news search
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')

            # Step 1: Collect recent news articles
            news_articles = await self._collect_news_articles(clean_symbol)

            # Step 2: Process each document with AI
            document_analyses = []
            for article in news_articles:
                try:
                    analysis = await self._analyze_document(article, 'news')
                    document_analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze article: {str(e)}")
                    continue

            # Step 3: Aggregate analysis results
            company_analysis = self._aggregate_document_analysis(symbol, document_analyses)

            logger.info(f"Document processing complete for {symbol}")
            return company_analysis

        except Exception as e:
            logger.error(f"Error processing documents for {symbol}: {str(e)}")
            raise

    async def _collect_news_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Collect recent news articles about the company"""
        articles = []

        try:
            # Get company info for better search
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            company_name = info.get('longName', symbol)

            # Simulate news collection (in production, use news APIs)
            # For demo, create sample articles
            sample_articles = [
                {
                    'title': f'{company_name} reports strong quarterly results',
                    'content': f'{company_name} announced strong quarterly earnings with revenue growth of 15% YoY. The company showed improved margins and robust cash flow generation.',
                    'url': f'https://example.com/news/{symbol}/earnings',
                    'published_date': datetime.now() - timedelta(days=2),
                    'source': 'Economic Times'
                },
                {
                    'title': f'{company_name} expands operations in new markets',
                    'content': f'{company_name} is expanding its operations into emerging markets as part of its growth strategy. The expansion is expected to drive future revenue growth.',
                    'url': f'https://example.com/news/{symbol}/expansion',
                    'published_date': datetime.now() - timedelta(days=5),
                    'source': 'Business Standard'
                },
                {
                    'title': f'Analysts upgrade {company_name} on strong fundamentals',
                    'content': f'Several analysts have upgraded their ratings on {company_name} citing strong fundamentals and improving market position.',
                    'url': f'https://example.com/news/{symbol}/upgrade',
                    'published_date': datetime.now() - timedelta(days=7),
                    'source': 'Money Control'
                }
            ]

            articles.extend(sample_articles)

            # In production, you would integrate with:
            # - News APIs (NewsAPI, Alpha Vantage News, etc.)
            # - Company investor relations pages
            # - SEC EDGAR filings (for US stocks)
            # - BSE/NSE announcements (for Indian stocks)

            logger.info(f"Collected {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"Error collecting news: {str(e)}")
            return []

    async def _analyze_document(self, document: Dict[str, Any], doc_type: str) -> DocumentAnalysis:
        """Analyze individual document with AI"""

        content = document.get('content', '')
        title = document.get('title', '')
        url = document.get('url', '')

        # Create analysis prompt
        analysis_prompt = f"""
        Analyze this {doc_type} document for investment insights:

        TITLE: {title}

        CONTENT: {content}

        Please provide:
        1. Sentiment Score (-1 to +1): Overall sentiment toward the company
        2. Key Themes: 3-5 main themes or topics discussed
        3. Financial Highlights: Any financial metrics, guidance, or performance mentioned
        4. Risk Factors: Any risks, challenges, or concerns mentioned
        5. Management Outlook: Summary of management's forward-looking statements
        6. Confidence Level (0-100): How confident are you in this analysis?

        Focus on information relevant for fundamental analysis and investment decisions.
        """

        try:
            # Get AI analysis
            ai_response = await self.ai_engine.get_completion(analysis_prompt)

            # Parse response (simplified for demo)
            sentiment_score = self._extract_sentiment_score(content, ai_response)
            key_themes = self._extract_key_themes(content, ai_response)
            financial_highlights = self._extract_financial_highlights(content)
            risk_factors = self._extract_risk_factors(content)
            management_outlook = self._extract_management_outlook(content)

            analysis = DocumentAnalysis(
                document_type=doc_type,
                sentiment_score=sentiment_score,
                key_themes=key_themes,
                financial_highlights=financial_highlights,
                risk_factors=risk_factors,
                management_outlook=management_outlook,
                confidence_level=75.0,  # Default confidence
                processing_timestamp=datetime.now(),
                source_url=url
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            # Return basic analysis
            return self._create_basic_analysis(document, doc_type)

    def _extract_sentiment_score(self, content: str, ai_response: str) -> float:
        """Extract sentiment score from content and AI response"""

        # Simple rule-based sentiment analysis as backup
        positive_count = sum(1 for word in self.sentiment_keywords['positive']
                             if word.lower() in content.lower())
        negative_count = sum(1 for word in self.sentiment_keywords['negative']
                             if word.lower() in content.lower())

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        sentiment_score = (positive_count - negative_count) / total_sentiment_words

        # Normalize to -1 to +1 range
        return max(-1.0, min(1.0, sentiment_score))

    def _extract_key_themes(self, content: str, ai_response: str) -> List[str]:
        """Extract key themes from document"""

        # Simple keyword extraction (in production, use proper NLP)
        business_keywords = [
            'revenue growth', 'market expansion', 'digital transformation',
            'cost optimization', 'new products', 'partnerships',
            'regulatory changes', 'competition', 'innovation'
        ]

        themes = []
        for keyword in business_keywords:
            if keyword.lower() in content.lower():
                themes.append(keyword.title())

        # Return top 5 themes
        return themes[:5] if themes else ['General business update']

    def _extract_financial_highlights(self, content: str) -> List[str]:
        """Extract financial metrics and highlights"""
        highlights = []

        # Look for financial patterns
        financial_patterns = [
            r'revenue.*?(\d+\.?\d*%)',
            r'profit.*?(\d+\.?\d*%)',
            r'growth.*?(\d+\.?\d*%)',
            r'margin.*?(\d+\.?\d*%)',
            r'EBITDA.*?(\d+\.?\d*)'
        ]

        for pattern in financial_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                highlights.append(match.group(0))

        return highlights[:5] if highlights else ['No specific financial metrics mentioned']

    def _extract_risk_factors(self, content: str) -> List[str]:
        """Extract risk factors mentioned"""
        risks = []

        risk_indicators = [
            'risk', 'challenge', 'concern', 'uncertainty', 'volatility',
            'competition', 'regulatory', 'market conditions'
        ]

        sentences = content.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in risk_indicators):
                risks.append(sentence.strip())

        return risks[:3] if risks else ['Standard business risks apply']

    def _extract_management_outlook(self, content: str) -> str:
        """Extract management outlook and forward guidance"""

        outlook_keywords = [
            'outlook', 'guidance', 'expect', 'forecast', 'future', 'plan'
        ]

        sentences = content.split('.')
        outlook_sentences = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in outlook_keywords):
                outlook_sentences.append(sentence.strip())

        if outlook_sentences:
            return '. '.join(outlook_sentences[:2])  # First 2 relevant sentences
        else:
            return 'No specific forward guidance provided'

    def _create_basic_analysis(self, document: Dict[str, Any], doc_type: str) -> DocumentAnalysis:
        """Create basic analysis when AI processing fails"""

        content = document.get('content', '')

        return DocumentAnalysis(
            document_type=doc_type,
            sentiment_score=self._extract_sentiment_score(content, ''),
            key_themes=['General business update'],
            financial_highlights=['No specific metrics available'],
            risk_factors=['Standard business risks'],
            management_outlook='No specific outlook provided',
            confidence_level=50.0,  # Lower confidence for basic analysis
            processing_timestamp=datetime.now(),
            source_url=document.get('url', '')
        )

    def _aggregate_document_analysis(self,
                                     symbol: str,
                                     document_analyses: List[DocumentAnalysis]) -> CompanyNewsAnalysis:
        """Aggregate individual document analyses into company-level insights"""

        if not document_analyses:
            # Return neutral analysis if no documents
            return CompanyNewsAnalysis(
                symbol=symbol,
                overall_sentiment=0.0,
                news_volume=0,
                key_developments=['No recent news available'],
                analyst_sentiment='NEUTRAL',
                document_analyses=[],
                summary='No recent news or documents available for analysis',
                last_updated=datetime.now()
            )

        # Calculate overall sentiment
        sentiment_scores = [doc.sentiment_score for doc in document_analyses]
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

        # Determine analyst sentiment
        if overall_sentiment > 0.2:
            analyst_sentiment = 'POSITIVE'
        elif overall_sentiment < -0.2:
            analyst_sentiment = 'NEGATIVE'
        else:
            analyst_sentiment = 'NEUTRAL'

        # Aggregate key developments
        all_themes = []
        for doc in document_analyses:
            all_themes.extend(doc.key_themes)

        # Count theme frequency
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Get top themes as key developments
        key_developments = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        key_developments = [theme for theme, count in key_developments[:5]]

        if not key_developments:
            key_developments = ['General business updates']

        # Create summary
        summary = self._create_news_summary(symbol, overall_sentiment, key_developments, len(document_analyses))

        return CompanyNewsAnalysis(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            news_volume=len(document_analyses),
            key_developments=key_developments,
            analyst_sentiment=analyst_sentiment,
            document_analyses=document_analyses,
            summary=summary,
            last_updated=datetime.now()
        )

    def _create_news_summary(self,
                             symbol: str,
                             sentiment: float,
                             key_developments: List[str],
                             article_count: int) -> str:
        """Create summary of news analysis"""

        sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"

        summary = f"""
        Analysis of {article_count} recent articles about {symbol} shows {sentiment_desc} sentiment 
        (score: {sentiment:.2f}). Key developments include: {', '.join(key_developments[:3])}. 
        This news analysis should be considered alongside fundamental and technical analysis 
        for comprehensive investment decisions.
        """.strip()

        return summary

    def integrate_with_fundamental_analysis(self,
                                            news_analysis: CompanyNewsAnalysis,
                                            fundamental_score: float) -> Dict[str, Any]:
        """
        Integrate news analysis with fundamental analysis
        Returns enhanced investment insights
        """

        # News sentiment impact on fundamental score
        news_impact = news_analysis.overall_sentiment * 10  # Scale to 0-10 impact

        # Adjust fundamental score based on news sentiment
        adjusted_score = fundamental_score + news_impact
        adjusted_score = max(0, min(100, adjusted_score))  # Keep in 0-100 range

        # Determine news-based risk level
        if abs(news_analysis.overall_sentiment) > 0.5:
            news_risk = "HIGH" if news_analysis.overall_sentiment < 0 else "LOW"
        else:
            news_risk = "MEDIUM"

        # Create integrated insights
        integration_result = {
            'original_fundamental_score': fundamental_score,
            'news_adjusted_score': adjusted_score,
            'news_sentiment_impact': news_impact,
            'news_sentiment': news_analysis.analyst_sentiment,
            'news_risk_level': news_risk,
            'key_news_themes': news_analysis.key_developments,
            'news_volume': news_analysis.news_volume,
            'integration_confidence': min(75.0, fundamental_score * 0.7 + abs(news_analysis.overall_sentiment) * 30),
            'recommendation_modifier': self._get_recommendation_modifier(news_analysis.overall_sentiment),
            'summary': news_analysis.summary
        }

        return integration_result

    def _get_recommendation_modifier(self, sentiment: float) -> str:
        """Get recommendation modifier based on news sentiment"""
        if sentiment > 0.3:
            return "NEWS_POSITIVE_CATALYST"
        elif sentiment < -0.3:
            return "NEWS_NEGATIVE_HEADWIND"
        else:
            return "NEWS_NEUTRAL"


# Example usage and testing
async def test_document_processor():
    """Test the document processor"""
    print("🧪 Testing AI Document Processor...")

    # Initialize processor
    processor = AIDocumentProcessor()

    # Test symbols
    test_symbols = ['RELIANCE', 'TCS', 'INFY']

    for symbol in test_symbols:
        try:
            print(f"\n📰 Processing documents for {symbol}...")
            news_analysis = await processor.process_company_documents(symbol)

            print(f"✅ Document Analysis for {symbol}:")
            print(f"   Overall Sentiment: {news_analysis.overall_sentiment:.2f}")
            print(f"   Analyst Sentiment: {news_analysis.analyst_sentiment}")
            print(f"   News Volume: {news_analysis.news_volume}")
            print(f"   Key Developments: {', '.join(news_analysis.key_developments[:3])}")
            print(f"   Summary: {news_analysis.summary[:100]}...")

            # Test integration with fundamental score
            sample_fundamental_score = 75.0
            integration = processor.integrate_with_fundamental_analysis(news_analysis, sample_fundamental_score)

            print(f"\n🔗 Integration Results:")
            print(f"   Original Score: {integration['original_fundamental_score']}")
            print(f"   News Adjusted: {integration['news_adjusted_score']:.1f}")
            print(f"   News Impact: {integration['news_sentiment_impact']:+.1f}")
            print(f"   Recommendation Modifier: {integration['recommendation_modifier']}")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_document_processor())