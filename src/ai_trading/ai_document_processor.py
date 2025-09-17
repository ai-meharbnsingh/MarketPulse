# src/ai_trading/ai_document_processor.py - FIXED VERSION

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import asyncio
import traceback
import re
from textblob import TextBlob


@dataclass
class DocumentAnalysis:
    """Document analysis result"""
    content: str
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # POSITIVE, NEGATIVE, NEUTRAL
    key_themes: List[str]
    financial_highlights: List[str]
    management_tone: str  # OPTIMISTIC, NEUTRAL, PESSIMISTIC
    confidence_score: float  # 0-1
    timestamp: datetime


@dataclass
class NewsAnalysis:
    """News article analysis"""
    symbol: str
    headline: str
    content: str
    source: str
    published_date: datetime
    sentiment_score: float
    impact_score: float  # 0-1
    relevance_score: float  # 0-1
    key_points: List[str]
    timestamp: datetime


class AIDocumentProcessor:
    """FIXED: AI-powered document and news analysis system"""

    def __init__(self):  # REMOVED ai_framework parameter
        """Initialize the document processor"""
        # Sentiment analysis thresholds
        self.sentiment_thresholds = {
            'very_positive': 0.5,
            'positive': 0.1,
            'neutral': 0.0,
            'negative': -0.1,
            'very_negative': -0.5
        }

        # Financial keywords for relevance scoring
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'acquisition',
            'merger', 'expansion', 'growth', 'decline', 'forecast', 'guidance',
            'quarterly', 'annual', 'results', 'performance', 'investment',
            'debt', 'equity', 'cash flow', 'margin', 'ebitda', 'eps'
        ]

        # Management tone indicators
        self.optimistic_words = [
            'confident', 'optimistic', 'positive', 'growth', 'expansion',
            'opportunity', 'strong', 'improve', 'increase', 'better'
        ]

        self.pessimistic_words = [
            'challenging', 'difficult', 'decline', 'decrease', 'concern',
            'uncertainty', 'risk', 'pressure', 'weak', 'lower'
        ]

    async def analyze_document(self, content: str, document_type: str = "general") -> DocumentAnalysis:
        """Comprehensive document analysis"""
        print(f"📄 Analyzing {document_type} document ({len(content)} characters)")

        try:
            # Basic sentiment analysis
            sentiment_score, sentiment_label = self._analyze_sentiment(content)

            # Extract key themes
            key_themes = self._extract_themes(content)

            # Extract financial highlights
            financial_highlights = self._extract_financial_highlights(content)

            # Analyze management tone
            management_tone = self._analyze_management_tone(content)

            # Calculate confidence score
            confidence_score = self._calculate_confidence(content, sentiment_score)

            return DocumentAnalysis(
                content=content[:500] + "..." if len(content) > 500 else content,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                key_themes=key_themes,
                financial_highlights=financial_highlights,
                management_tone=management_tone,
                confidence_score=confidence_score,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Error analyzing document: {e}")
            return self._create_default_analysis(content)

    async def analyze_news_batch(self, symbol: str, articles: List[Dict]) -> List[NewsAnalysis]:
        """Analyze multiple news articles for a symbol"""
        print(f"📰 Analyzing {len(articles)} news articles for {symbol}")

        analyses = []

        for article in articles:
            try:
                analysis = await self.analyze_news_article(symbol, article)
                analyses.append(analysis)
            except Exception as e:
                print(f"⚠️ Error analyzing article: {e}")
                continue

        return analyses

    async def analyze_news_article(self, symbol: str, article: Dict) -> NewsAnalysis:
        """Analyze individual news article"""

        try:
            headline = article.get('headline', article.get('title', ''))
            content = article.get('content', article.get('summary', ''))
            source = article.get('source', 'Unknown')
            published_date = article.get('published_date', datetime.now())

            # Combine headline and content for analysis
            full_text = f"{headline} {content}"

            # Sentiment analysis
            sentiment_score, _ = self._analyze_sentiment(full_text)

            # Calculate impact and relevance scores
            impact_score = self._calculate_impact_score(headline, content)
            relevance_score = self._calculate_relevance_score(symbol, full_text)

            # Extract key points
            key_points = self._extract_key_points(content)

            return NewsAnalysis(
                symbol=symbol,
                headline=headline,
                content=content[:300] + "..." if len(content) > 300 else content,
                source=source,
                published_date=published_date,
                sentiment_score=sentiment_score,
                impact_score=impact_score,
                relevance_score=relevance_score,
                key_points=key_points,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Error analyzing news article: {e}")
            return self._create_default_news_analysis(symbol)

    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            # Classify sentiment
            if polarity >= self.sentiment_thresholds['very_positive']:
                label = "VERY_POSITIVE"
            elif polarity >= self.sentiment_thresholds['positive']:
                label = "POSITIVE"
            elif polarity <= self.sentiment_thresholds['very_negative']:
                label = "VERY_NEGATIVE"
            elif polarity <= self.sentiment_thresholds['negative']:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            return polarity, label

        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return 0.0, "NEUTRAL"

    def _extract_themes(self, content: str) -> List[str]:
        """Extract key themes from content"""
        try:
            # Simple keyword-based theme extraction
            content_lower = content.lower()
            themes = []

            theme_keywords = {
                'Earnings & Results': ['earnings', 'quarterly', 'annual', 'results', 'performance'],
                'Growth & Expansion': ['growth', 'expansion', 'increase', 'rising', 'growing'],
                'Financial Health': ['debt', 'cash', 'margin', 'profit', 'revenue'],
                'Market Position': ['market', 'competition', 'share', 'leadership'],
                'Strategy & Vision': ['strategy', 'vision', 'plan', 'future', 'direction'],
                'Risk & Challenges': ['risk', 'challenge', 'concern', 'uncertainty', 'pressure'],
                'Innovation & Technology': ['technology', 'innovation', 'digital', 'ai', 'automation'],
                'Regulatory & Compliance': ['regulatory', 'compliance', 'policy', 'regulation']
            }

            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    themes.append(theme)

            return themes[:5]  # Return top 5 themes

        except Exception as e:
            print(f"⚠️ Theme extraction error: {e}")
            return ["General Business"]

    def _extract_financial_highlights(self, content: str) -> List[str]:
        """Extract financial highlights and numbers"""
        try:
            highlights = []

            # Look for percentage mentions
            percent_pattern = r'(\d+(?:\.\d+)?%)'
            percentages = re.findall(percent_pattern, content)
            for pct in percentages[:3]:  # Top 3 percentages
                highlights.append(f"Percentage mentioned: {pct}")

            # Look for currency amounts (₹, $, etc.)
            currency_pattern = r'([₹$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion|trillion))?)'
            amounts = re.findall(currency_pattern, content, re.IGNORECASE)
            for amount in amounts[:3]:  # Top 3 amounts
                highlights.append(f"Amount mentioned: {amount}")

            # Look for growth/decline mentions
            growth_pattern = r'((?:increased|decreased|grew|declined|rose|fell)\s+by\s+\d+(?:\.\d+)?%)'
            growth_mentions = re.findall(growth_pattern, content, re.IGNORECASE)
            for growth in growth_mentions[:2]:
                highlights.append(f"Performance: {growth}")

            return highlights[:5] if highlights else ["Financial data analysis in progress"]

        except Exception as e:
            print(f"⚠️ Financial highlights extraction error: {e}")
            return ["Financial analysis available"]

    def _analyze_management_tone(self, content: str) -> str:
        """Analyze management tone from content"""
        try:
            content_lower = content.lower()

            optimistic_count = sum(1 for word in self.optimistic_words if word in content_lower)
            pessimistic_count = sum(1 for word in self.pessimistic_words if word in content_lower)

            if optimistic_count > pessimistic_count * 1.5:
                return "OPTIMISTIC"
            elif pessimistic_count > optimistic_count * 1.5:
                return "PESSIMISTIC"
            else:
                return "NEUTRAL"

        except Exception as e:
            print(f"⚠️ Management tone analysis error: {e}")
            return "NEUTRAL"

    def _calculate_confidence(self, content: str, sentiment_score: float) -> float:
        """Calculate confidence score for analysis"""
        try:
            # Base confidence on content length and sentiment clarity
            length_factor = min(len(content) / 1000, 1.0)  # Max factor of 1.0
            sentiment_clarity = abs(sentiment_score)  # Higher absolute value = clearer sentiment

            # Check for financial keywords
            content_lower = content.lower()
            keyword_count = sum(1 for keyword in self.financial_keywords if keyword in content_lower)
            keyword_factor = min(keyword_count / 10, 1.0)  # Max factor of 1.0

            confidence = (length_factor * 0.4 + sentiment_clarity * 0.4 + keyword_factor * 0.2)
            return min(confidence, 0.95)  # Cap at 95%

        except Exception as e:
            print(f"⚠️ Confidence calculation error: {e}")
            return 0.5

    def _calculate_impact_score(self, headline: str, content: str) -> float:
        """Calculate news impact score"""
        try:
            # Combine headline and content
            full_text = f"{headline} {content}".lower()

            # High impact keywords
            high_impact_words = [
                'acquisition', 'merger', 'bankruptcy', 'lawsuit', 'regulation',
                'guidance', 'forecast', 'earnings', 'dividend', 'split'
            ]

            # Medium impact keywords
            medium_impact_words = [
                'partnership', 'expansion', 'growth', 'decline', 'increase',
                'decrease', 'appointment', 'resignation', 'investment'
            ]

            score = 0.3  # Base score

            for word in high_impact_words:
                if word in full_text:
                    score += 0.2

            for word in medium_impact_words:
                if word in full_text:
                    score += 0.1

            return min(score, 1.0)

        except Exception as e:
            print(f"⚠️ Impact score calculation error: {e}")
            return 0.5

    def _calculate_relevance_score(self, symbol: str, content: str) -> float:
        """Calculate relevance score for the symbol"""
        try:
            content_lower = content.lower()
            symbol_clean = symbol.replace('.NS', '').replace('.BO', '').lower()

            # Direct symbol mention
            if symbol_clean in content_lower:
                score = 0.8
            else:
                score = 0.4

            # Financial keywords boost relevance
            keyword_count = sum(1 for keyword in self.financial_keywords if keyword in content_lower)
            keyword_boost = min(keyword_count * 0.05, 0.2)

            return min(score + keyword_boost, 1.0)

        except Exception as e:
            print(f"⚠️ Relevance score calculation error: {e}")
            return 0.5

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        try:
            # Simple sentence-based extraction
            sentences = content.split('.')[:5]  # First 5 sentences
            key_points = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in self.financial_keywords):
                    key_points.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)

            return key_points[:3] if key_points else ["Analysis in progress"]

        except Exception as e:
            print(f"⚠️ Key points extraction error: {e}")
            return ["Content analysis available"]

    def _create_default_analysis(self, content: str) -> DocumentAnalysis:
        """Create default analysis when processing fails"""
        return DocumentAnalysis(
            content=content[:100] + "..." if len(content) > 100 else content,
            sentiment_score=0.0,
            sentiment_label="NEUTRAL",
            key_themes=["General Analysis"],
            financial_highlights=["Analysis in progress"],
            management_tone="NEUTRAL",
            confidence_score=0.3,
            timestamp=datetime.now()
        )

    def _create_default_news_analysis(self, symbol: str) -> NewsAnalysis:
        """Create default news analysis when processing fails"""
        return NewsAnalysis(
            symbol=symbol,
            headline="Analysis in progress",
            content="Content processing",
            source="Unknown",
            published_date=datetime.now(),
            sentiment_score=0.0,
            impact_score=0.3,
            relevance_score=0.5,
            key_points=["Processing news content"],
            timestamp=datetime.now()
        )

    def format_document_report(self, analysis: DocumentAnalysis) -> str:
        """Format document analysis into readable report"""

        report = f"""
📄 DOCUMENT ANALYSIS REPORT
{'=' * 40}
Analysis Time: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

📊 SENTIMENT ANALYSIS:
Score: {analysis.sentiment_score:.2f} ({analysis.sentiment_label})
Management Tone: {analysis.management_tone}
Confidence: {analysis.confidence_score:.1%}

🎯 KEY THEMES ({len(analysis.key_themes)}):
{chr(10).join(['• ' + theme for theme in analysis.key_themes])}

💰 FINANCIAL HIGHLIGHTS ({len(analysis.financial_highlights)}):
{chr(10).join(['• ' + highlight for highlight in analysis.financial_highlights])}

📝 CONTENT PREVIEW:
{analysis.content}
        """

        return report.strip()

    def format_news_summary(self, analyses: List[NewsAnalysis]) -> str:
        """Format news analyses into summary report"""

        if not analyses:
            return "No news analyses available"

        symbol = analyses[0].symbol
        avg_sentiment = np.mean([a.sentiment_score for a in analyses])
        avg_impact = np.mean([a.impact_score for a in analyses])

        sentiment_label = "POSITIVE" if avg_sentiment > 0.1 else "NEGATIVE" if avg_sentiment < -0.1 else "NEUTRAL"

        report = f"""
📰 NEWS SENTIMENT SUMMARY - {symbol}
{'=' * 40}
Articles Analyzed: {len(analyses)}
Average Sentiment: {avg_sentiment:.2f} ({sentiment_label})
Average Impact Score: {avg_impact:.2f}

📊 TOP ARTICLES:
"""

        # Sort by impact score and show top 3
        top_articles = sorted(analyses, key=lambda x: x.impact_score, reverse=True)[:3]

        for i, article in enumerate(top_articles, 1):
            report += f"""
{i}. {article.headline}
   Source: {article.source}
   Sentiment: {article.sentiment_score:.2f}
   Impact: {article.impact_score:.2f}
   Key Points: {article.key_points[0] if article.key_points else 'Processing...'}
"""

        return report.strip()


# Testing function
async def test_document_processor():
    """Test the document processor"""
    print("🧪 Testing AI Document Processor")
    print("=" * 50)

    processor = AIDocumentProcessor()

    # Test document analysis
    sample_document = """
    Q2 FY2024 results show strong performance with revenue growing 15% year-over-year. 
    The company reported earnings per share of ₹45, beating analyst expectations of ₹42.
    Management expressed confidence in achieving full-year guidance of 12-15% growth.
    New product launches and expanding market presence are key drivers.
    However, rising input costs remain a challenge for margins.
    """

    print("📄 Testing document analysis...")
    analysis = await processor.analyze_document(sample_document, "earnings_report")

    print(f"✅ Document analysis complete")
    print(f"Sentiment: {analysis.sentiment_label} ({analysis.sentiment_score:.2f})")
    print(f"Themes: {analysis.key_themes}")
    print(f"Management Tone: {analysis.management_tone}")

    # Print full report
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS REPORT:")
    print("=" * 50)
    print(processor.format_document_report(analysis))
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_document_processor())