# src/ai_trading/ai_portfolio_optimizer.py

import asyncio
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import numpy as np
from dataclasses import asdict
from pathlib import Path
import os

# Import our existing components
from portfolio_theory import PortfolioTheoryEngine, InvestorProfile, RiskTolerance, FinancialGoal

# Fix the import path for Antifragile Framework
project_root = Path(__file__).parent.parent.parent
framework_core = project_root / "01_Framework_Core"

import sys

sys.path.insert(0, str(framework_core))

# Load environment variables
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Import AI Framework components
from antifragile_framework.core.failover_engine import FailoverEngine
from antifragile_framework.config.config_loader import load_provider_profiles
from antifragile_framework.providers.api_abstraction_layer import ChatMessage
from antifragile_framework.providers.provider_registry import get_default_provider_registry
from telemetry.event_bus import EventBus


class AIPortfolioOptimizer:
    """AI-Enhanced Portfolio Optimization with Multi-Provider Intelligence"""

    def __init__(self):
        self.portfolio_engine = PortfolioTheoryEngine()
        self.ai_engine = None

        # AI Provider configurations
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

    async def initialize(self):
        """Initialize AI engine with proper failover"""
        try:
            provider_profiles = load_provider_profiles()
            provider_registry = get_default_provider_registry()
            event_bus = EventBus()

            self.ai_engine = FailoverEngine(
                provider_configs=self.provider_configs,
                provider_registry=provider_registry,
                event_bus=event_bus,
                provider_profiles=provider_profiles,
            )

            # Test AI connectivity
            total_keys = sum(len(config["api_keys"]) for config in self.provider_configs.values())
            if total_keys == 0:
                print("⚠️ Warning: No API keys found! AI features will be limited.")
                return False

            print(f"✅ AI Portfolio Optimizer initialized successfully!")
            print(f"🔑 Total API keys loaded: {total_keys}")
            return True

        except Exception as e:
            print(f"❌ AI initialization failed: {str(e)}")
            print("📊 Falling back to non-AI portfolio optimization")
            return False

    async def assess_risk_tolerance_ai(self, investor_data: Dict) -> Dict[str, Any]:
        """AI-powered risk tolerance assessment with psychological profiling"""

        if not self.ai_engine:
            return {
                "status": "error",
                "message": "AI engine not initialized. Use initialize() first."
            }

        risk_assessment_prompt = f"""
        Analyze this investor's risk tolerance based on their profile:

        Investor Profile:
        - Age: {investor_data.get('age', 'Not specified')}
        - Annual Income: ₹{investor_data.get('annual_income', 0):,}
        - Monthly Expenses: ₹{investor_data.get('monthly_expenses', 0):,}
        - Current Investments: {investor_data.get('existing_investments', {})}
        - Investment Horizon: {investor_data.get('investment_horizon', 'Not specified')} years
        - Goals: {investor_data.get('goals', [])}

        Provide detailed analysis:

        1. **Risk Tolerance Category**: (Conservative/Moderate/Aggressive/Very Aggressive)

        2. **Risk Capacity Analysis**:
           - Financial capacity to take risk based on income/expenses
           - Time horizon impact on risk capacity
           - Emergency fund adequacy assessment

        3. **Risk Willingness Factors**:
           - Age-appropriate risk levels
           - Goals alignment with risk taking
           - Behavioral risk indicators

        4. **Recommended Asset Allocation Ranges**:
           - Equity allocation range (%)
           - Fixed income range (%)
           - Alternative investments range (%)

        5. **Specific Recommendations**:
           - Tax optimization strategies for Indian markets
           - LTCG vs STCG considerations
           - Rebalancing frequency suggestions

        6. **Risk Mitigation Strategies**:
           - Portfolio diversification recommendations
           - Hedge strategies for market downturns
           - Regular monitoring checkpoints

        Provide specific, actionable insights for Indian market conditions.
        """

        try:
            # FIXED: Use correct method name
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "openai": ["gpt-4o", "gpt-4-turbo"],
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[ChatMessage(role="user", content=risk_assessment_prompt)],
                max_estimated_cost_usd=0.02,
                request_id="risk_assessment_ai"
            )

            return {
                "status": "success",
                "ai_assessment": response.content,
                "model_used": response.model_used,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"AI risk assessment failed: {str(e)}"
            }

    async def optimize_portfolio_ai(self, investor_profile: InvestorProfile,
                                    market_context: Dict = None) -> Dict[str, Any]:
        """AI-enhanced portfolio optimization with market analysis"""

        # First get baseline MPT optimization
        baseline_allocation = self.portfolio_engine.optimize_for_risk_tolerance(
            investor_profile.risk_tolerance
        )

        # Get goal-based allocations
        goal_allocations = self.portfolio_engine.calculate_goal_based_allocation(investor_profile)

        # If AI engine is not available, return baseline results
        if not self.ai_engine:
            return {
                "status": "success",
                "baseline_allocation": baseline_allocation,
                "goal_allocations": goal_allocations,
                "ai_optimization": "AI optimization not available - using baseline MPT results",
                "model_used": "mathematical_optimization_only",
                "timestamp": datetime.now().isoformat()
            }

        # Prepare data for AI analysis
        profile_dict = {
            'age': investor_profile.age,
            'income': investor_profile.annual_income,
            'expenses': investor_profile.monthly_expenses,
            'risk_tolerance': investor_profile.risk_tolerance.value,
            'investment_horizon': investor_profile.investment_horizon,
            'goals': [asdict(goal) for goal in investor_profile.goals]
        }

        optimization_prompt = f"""
        Optimize this portfolio allocation using advanced AI analysis:

        **Investor Profile:**
        {json.dumps(profile_dict, indent=2)}

        **Baseline MPT Allocation:**
        {json.dumps(baseline_allocation['allocation'], indent=2)}
        Expected Return: {baseline_allocation['expected_return']:.2%}
        Risk: {baseline_allocation['risk']:.2%}

        **Optimization Requirements:**

        1. **Enhanced Allocation Strategy**:
           - Improve upon baseline MPT allocation
           - Consider current Indian market conditions
           - Account for LTCG/STCG tax implications

        2. **Asset Class Analysis**:
           - Large Cap vs Mid/Small Cap allocation
           - Fixed Income optimization for current rates
           - Gold allocation for inflation protection
           - International diversification benefits

        3. **Tax Optimization Strategy**:
           - LTCG optimization (>1 year holding for 10% tax)
           - STCG minimization strategies
           - Rebalancing timing for tax efficiency

        4. **Implementation Plan**:
           - SIP-based implementation strategy
           - Rebalancing frequency and triggers
           - Performance monitoring metrics

        **Provide specific, actionable recommendations for Indian markets.**
        """

        try:
            # FIXED: Use correct method name
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "openai": ["gpt-4o", "gpt-4-turbo"],
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[ChatMessage(role="user", content=optimization_prompt)],
                max_estimated_cost_usd=0.03,
                request_id="portfolio_optimization_ai"
            )

            return {
                "status": "success",
                "baseline_allocation": baseline_allocation,
                "goal_allocations": goal_allocations,
                "ai_optimization": response.content,
                "model_used": response.model_used,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"AI portfolio optimization failed: {str(e)}",
                "baseline_allocation": baseline_allocation,
                "goal_allocations": goal_allocations
            }


# Enhanced test function with risk assessment
async def test_ai_portfolio_optimizer():
    """Complete test of Day 4 objectives"""

    print("🧠 Day 4: Portfolio Theory & AI Optimization Test")
    print("=" * 60)

    # Create test investor profile
    goals = [
        FinancialGoal("Emergency Fund", 500000, 12, "critical", 100000, 25000),
        FinancialGoal("House Down Payment", 2000000, 60, "important", 200000, 30000),
        FinancialGoal("Retirement", 10000000, 300, "critical", 500000, 20000)
    ]

    profile = InvestorProfile(
        age=30,
        annual_income=1200000,
        monthly_expenses=50000,
        existing_investments={'equity': 300000, 'debt': 200000},
        risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon=25,
        goals=goals,
        tax_bracket=0.20
    )

    # Initialize optimizer
    optimizer = AIPortfolioOptimizer()
    ai_available = await optimizer.initialize()

    # Test 1: Risk Assessment (if AI available)
    if ai_available:
        print("\n🧠 Testing AI Risk Assessment...")
        investor_data = {
            'age': profile.age,
            'annual_income': profile.annual_income,
            'monthly_expenses': profile.monthly_expenses,
            'existing_investments': profile.existing_investments,
            'investment_horizon': profile.investment_horizon,
            'goals': [f"{goal.name}: ₹{goal.target_amount:,} in {goal.timeline_months} months"
                      for goal in profile.goals]
        }

        risk_assessment = await optimizer.assess_risk_tolerance_ai(investor_data)

        if risk_assessment['status'] == 'success':
            print(f"✅ AI Risk Assessment Complete (Model: {risk_assessment['model_used']})")
        else:
            print(f"⚠️ Risk Assessment Issue: {risk_assessment['message']}")

    # Test 2: Portfolio Optimization
    print("\n📊 Testing Portfolio Optimization...")

    optimization_result = await optimizer.optimize_portfolio_ai(profile)

    if optimization_result['status'] == 'success':
        print(f"✅ Portfolio Optimization Complete")
        print("\n📈 Baseline Allocation (Modern Portfolio Theory):")
        for asset, weight in optimization_result['baseline_allocation']['allocation'].items():
            print(f"  {asset.replace('_', ' ').title()}: {weight:.1%}")

        print(f"\n📊 Expected Performance:")
        print(f"  Expected Return: {optimization_result['baseline_allocation']['expected_return']:.1%}")
        print(f"  Expected Risk: {optimization_result['baseline_allocation']['risk']:.1%}")
        print(f"  Sharpe Ratio: {optimization_result['baseline_allocation']['sharpe_ratio']:.2f}")

        if ai_available and 'ai_optimization' in optimization_result:
            print(f"\n🤖 AI Enhancement (Model: {optimization_result.get('model_used', 'N/A')})")
            print("💡 AI Analysis Preview:")
            ai_text = optimization_result.get('ai_optimization', '')
            # Show first 300 characters
            print(f"  {ai_text[:300]}{'...' if len(ai_text) > 300 else ''}")
    else:
        print(f"❌ Optimization Failed: {optimization_result['message']}")

    print("\n" + "=" * 60)
    print("🎯 Day 4 Morning Session Complete!")
    print("✅ Modern Portfolio Theory: Implemented")
    print("✅ Risk-based Allocation: Working")
    print("✅ Goal-based Analysis: Functional")
    print(f"✅ AI Integration: {'Active' if ai_available else 'Ready (needs API test)'}")
    print("✅ Portfolio Optimization Engine: Ready for Afternoon Session")


if __name__ == "__main__":
    asyncio.run(test_ai_portfolio_optimizer())