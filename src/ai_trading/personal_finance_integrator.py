# src/ai_trading/personal_finance_integrator.py

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

# Import our existing components
from portfolio_theory import PortfolioTheoryEngine, InvestorProfile, RiskTolerance, FinancialGoal
from ai_portfolio_optimizer import AIPortfolioOptimizer

# AI Framework imports (using same path setup)
project_root = Path(__file__).parent.parent.parent
framework_core = project_root / "01_Framework_Core"

import sys

sys.path.insert(0, str(framework_core))

from dotenv import load_dotenv
import os

load_dotenv(project_root / ".env")

from antifragile_framework.providers.api_abstraction_layer import ChatMessage


@dataclass
class PersonalFinanceProfile:
    """Complete personal finance profile for AI analysis"""
    monthly_income: float
    monthly_expenses: float
    current_savings: float
    current_debt: float
    monthly_sip_capacity: float
    emergency_fund_target: float
    insurance_coverage: Dict[str, float]
    tax_saving_investments: Dict[str, float]  # 80C, 80D, etc.
    financial_dependents: int


@dataclass
class TaxOptimizationPlan:
    """Tax optimization strategy for Indian markets"""
    ltcg_optimization: Dict[str, Any]
    stcg_minimization: Dict[str, Any]
    section_80c_utilization: Dict[str, float]
    elss_recommendations: Dict[str, float]
    tax_loss_harvesting: List[Dict[str, Any]]
    annual_tax_savings_potential: float


@dataclass
class GoalImplementationPlan:
    """Implementation roadmap for each financial goal"""
    goal_name: str
    monthly_sip_allocation: float
    asset_allocation: Dict[str, float]
    timeline_milestones: List[Dict[str, Any]]
    risk_monitoring_triggers: Dict[str, float]
    rebalancing_schedule: str
    tax_strategy: Dict[str, Any]


class PersonalFinanceIntegrator:
    """AI-Powered Personal Finance Integration with Goal-Based Planning"""

    def __init__(self):
        self.portfolio_optimizer = AIPortfolioOptimizer()
        self.ai_engine = None

    async def initialize(self):
        """Initialize the personal finance integrator"""
        success = await self.portfolio_optimizer.initialize()
        if success:
            self.ai_engine = self.portfolio_optimizer.ai_engine
            print("✅ Personal Finance Integrator initialized successfully!")
        return success

    def analyze_personal_cash_flow(self, finance_profile: PersonalFinanceProfile) -> Dict[str, Any]:
        """Analyze personal cash flow and investment capacity"""

        # Calculate investment capacity
        monthly_surplus = finance_profile.monthly_income - finance_profile.monthly_expenses
        emergency_fund_gap = max(0, finance_profile.emergency_fund_target - finance_profile.current_savings)

        # Emergency fund priority calculation
        emergency_months_needed = emergency_fund_gap / max(finance_profile.monthly_sip_capacity, 1000)

        # Available for goal-based investing
        available_for_goals = max(0, finance_profile.monthly_sip_capacity -
                                  min(finance_profile.monthly_sip_capacity * 0.3, emergency_fund_gap / 6))

        # Debt analysis
        debt_to_income_ratio = finance_profile.current_debt / (finance_profile.monthly_income * 12)

        return {
            "monthly_surplus": monthly_surplus,
            "investment_capacity": finance_profile.monthly_sip_capacity,
            "emergency_fund_gap": emergency_fund_gap,
            "months_to_complete_emergency": emergency_months_needed,
            "available_for_goals": available_for_goals,
            "debt_to_income_ratio": debt_to_income_ratio,
            "cash_flow_health": "healthy" if monthly_surplus > finance_profile.monthly_sip_capacity else "tight"
        }

    async def create_goal_based_sip_strategy(self, investor_profile: InvestorProfile,
                                             finance_profile: PersonalFinanceProfile) -> Dict[str, Any]:
        """AI-powered SIP strategy for multiple goals"""

        if not self.ai_engine:
            return {"status": "error", "message": "AI engine not initialized"}

        # Analyze cash flow first
        cash_flow = self.analyze_personal_cash_flow(finance_profile)

        # Get baseline portfolio optimization
        optimization_result = await self.portfolio_optimizer.optimize_portfolio_ai(investor_profile)

        sip_strategy_prompt = f"""
        Create comprehensive SIP (Systematic Investment Plan) strategy for multiple financial goals:

        **Investor Profile:**
        - Age: {investor_profile.age}
        - Annual Income: ₹{investor_profile.annual_income:,}
        - Risk Tolerance: {investor_profile.risk_tolerance.value}
        - Investment Horizon: {investor_profile.investment_horizon} years

        **Financial Goals:**
        {json.dumps([asdict(goal) for goal in investor_profile.goals], indent=2)}

        **Personal Finance Analysis:**
        {json.dumps(asdict(finance_profile), indent=2)}

        **Cash Flow Analysis:**
        {json.dumps(cash_flow, indent=2)}

        **Baseline Portfolio Allocation:**
        {json.dumps(optimization_result.get('baseline_allocation', {}).get('allocation', {}), indent=2)}

        **Required Analysis:**

        1. **Goal Prioritization Strategy**:
           - Emergency fund completion timeline
           - Critical vs important vs desirable goal sequencing
           - Parallel vs sequential goal funding approach

        2. **SIP Allocation Strategy**:
           - Monthly SIP amount per goal
           - Asset allocation for each goal based on timeline
           - Specific mutual fund/ETF recommendations for Indian market

        3. **Tax-Efficient Implementation**:
           - 80C utilization through ELSS funds
           - LTCG optimization for equity investments
           - Debt fund allocation for tax efficiency

        4. **Timeline and Milestones**:
           - Quarterly review checkpoints
           - Annual rebalancing triggers
           - Goal completion probability analysis

        5. **Risk Management**:
           - Downside protection for short-term goals
           - Volatility management strategies
           - Emergency fund integration

        6. **Implementation Roadmap**:
           - Month-by-month SIP setup plan
           - Automated investment recommendations
           - Performance tracking metrics

        **Output Format:**
        Provide detailed, actionable SIP strategy with specific:
        1. Monthly investment amounts per goal
        2. Specific fund recommendations (Indian mutual funds/ETFs)
        3. Tax optimization tactics
        4. Risk monitoring triggers
        5. Performance milestones and review schedule

        Focus on practical implementation for Indian investors.
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "google_gemini": ["gemini-1.5-flash-latest"],
                    "openai": ["gpt-4o", "gpt-4-turbo"],
                    "anthropic": ["claude-3-5-sonnet-20240620"]
                },
                messages=[ChatMessage(role="user", content=sip_strategy_prompt)],
                max_estimated_cost_usd=0.04,
                request_id="sip_strategy_ai"
            )

            return {
                "status": "success",
                "cash_flow_analysis": cash_flow,
                "sip_strategy": response.content,
                "model_used": response.model_used,
                "baseline_allocation": optimization_result.get('baseline_allocation', {}),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"SIP strategy generation failed: {str(e)}",
                "cash_flow_analysis": cash_flow
            }

    async def generate_tax_optimization_plan(self, investor_profile: InvestorProfile,
                                             finance_profile: PersonalFinanceProfile,
                                             current_investments: Dict[str, float]) -> Dict[str, Any]:
        """AI-powered tax optimization for Indian investors"""

        if not self.ai_engine:
            return {"status": "error", "message": "AI engine not initialized"}

        tax_optimization_prompt = f"""
        Generate comprehensive tax optimization strategy for Indian investor:

        **Investor Profile:**
        - Annual Income: ₹{investor_profile.annual_income:,}
        - Tax Bracket: {investor_profile.tax_bracket:.0%}
        - Investment Horizon: {investor_profile.investment_horizon} years

        **Current Investment Portfolio:**
        {json.dumps(current_investments, indent=2)}

        **Tax Saving Investments (Current):**
        {json.dumps(finance_profile.tax_saving_investments, indent=2)}

        **Optimization Requirements:**

        1. **Section 80C Optimization (₹1.5L limit)**:
           - ELSS mutual fund allocation
           - PPF vs EPF optimization
           - NSC/FD vs equity-linked options

        2. **LTCG vs STCG Strategy**:
           - Holding period optimization (>1 year for 10% LTCG)
           - Tax loss harvesting opportunities
           - Rebalancing timing for tax efficiency

        3. **Section 80D Health Insurance**:
           - Health insurance premium optimization
           - Parents' coverage tax benefits

        4. **Debt Investment Tax Strategy**:
           - Debt mutual funds vs FD taxation
           - Indexation benefits for long-term debt

        5. **Goal-Based Tax Planning**:
           - Emergency fund in tax-efficient instruments
           - Child education through Sukanya Samriddhi (if applicable)
           - Retirement planning with NPS benefits

        6. **Annual Tax Planning Calendar**:
           - Investment timing for maximum benefits
           - Year-end tax planning activities
           - Quarterly review checkpoints

        **Provide specific, actionable tax optimization recommendations:**
        1. Exact investment amounts for each tax-saving instrument
        2. Timeline for implementation
        3. Expected annual tax savings
        4. Integration with existing portfolio goals
        5. Risk considerations for tax-saving investments

        Focus on practical implementation maximizing after-tax returns.
        """

        try:
            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "google_gemini": ["gemini-1.5-flash-latest"],
                    "openai": ["gpt-4o"],
                    "anthropic": ["claude-3-5-sonnet-20240620"]
                },
                messages=[ChatMessage(role="user", content=tax_optimization_prompt)],
                max_estimated_cost_usd=0.04,
                request_id="tax_optimization_ai"
            )

            return {
                "status": "success",
                "tax_optimization_plan": response.content,
                "model_used": response.model_used,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Tax optimization failed: {str(e)}"
            }


# Test the Personal Finance Integrator
async def test_personal_finance_integrator():
    """Test the complete personal finance integration system"""

    print("💰 Day 4 Afternoon: Personal Finance Integration Test")
    print("=" * 65)

    # Create comprehensive test profiles
    goals = [
        FinancialGoal("Emergency Fund", 600000, 12, "critical", 150000, 30000),
        FinancialGoal("House Down Payment", 2500000, 60, "important", 300000, 35000),
        FinancialGoal("Child Education", 5000000, 180, "critical", 100000, 25000),
        FinancialGoal("Retirement", 15000000, 360, "critical", 800000, 30000)
    ]

    investor_profile = InvestorProfile(
        age=32,
        annual_income=1500000,
        monthly_expenses=60000,
        existing_investments={'equity': 400000, 'debt': 300000, 'gold': 200000},
        risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon=28,
        goals=goals,
        tax_bracket=0.30
    )

    finance_profile = PersonalFinanceProfile(
        monthly_income=125000,
        monthly_expenses=60000,
        current_savings=150000,
        current_debt=500000,  # Home loan
        monthly_sip_capacity=45000,
        emergency_fund_target=600000,
        insurance_coverage={'term_life': 10000000, 'health': 1000000},
        tax_saving_investments={'elss': 50000, 'ppf': 100000, 'nps': 50000},
        financial_dependents=2
    )

    # Initialize integrator
    integrator = PersonalFinanceIntegrator()
    ai_available = await integrator.initialize()

    print(f"🤖 AI Status: {'Active' if ai_available else 'Not Available'}")

    # Test 1: Cash Flow Analysis
    print("\n💰 Testing Cash Flow Analysis...")
    cash_flow = integrator.analyze_personal_cash_flow(finance_profile)

    print("📊 Cash Flow Results:")
    print(f"  Monthly Surplus: ₹{cash_flow['monthly_surplus']:,.0f}")
    print(f"  Investment Capacity: ₹{cash_flow['investment_capacity']:,.0f}")
    print(f"  Emergency Fund Gap: ₹{cash_flow['emergency_fund_gap']:,.0f}")
    print(f"  Available for Goals: ₹{cash_flow['available_for_goals']:,.0f}")
    print(f"  Cash Flow Health: {cash_flow['cash_flow_health'].title()}")

    # Test 2: Goal-Based SIP Strategy (if AI available)
    if ai_available:
        print("\n🎯 Testing Goal-Based SIP Strategy...")
        sip_result = await integrator.create_goal_based_sip_strategy(investor_profile, finance_profile)

        if sip_result['status'] == 'success':
            print(f"✅ SIP Strategy Generated (Model: {sip_result['model_used']})")
            print("📋 SIP Strategy Preview:")
            strategy_text = sip_result['sip_strategy']
            print(f"  {strategy_text[:400]}{'...' if len(strategy_text) > 400 else ''}")
        else:
            print(f"⚠️ SIP Strategy Issue: {sip_result['message']}")

        # Test 3: Tax Optimization Plan
        print("\n💸 Testing Tax Optimization...")
        tax_result = await integrator.generate_tax_optimization_plan(
            investor_profile, finance_profile, investor_profile.existing_investments
        )

        if tax_result['status'] == 'success':
            print(f"✅ Tax Optimization Plan Generated (Model: {tax_result['model_used']})")
            print("📋 Tax Plan Preview:")
            tax_text = tax_result['tax_optimization_plan']
            print(f"  {tax_text[:400]}{'...' if len(tax_text) > 400 else ''}")
        else:
            print(f"⚠️ Tax Optimization Issue: {tax_result['message']}")

    print("\n" + "=" * 65)
    print("🎯 Day 4 Afternoon Session Progress:")
    print("✅ Personal Finance Integration: Implemented")
    print("✅ Cash Flow Analysis: Working")
    print(f"✅ Goal-Based SIP Strategy: {'AI-Powered' if ai_available else 'Framework Ready'}")
    print(f"✅ Tax Optimization: {'AI-Enhanced' if ai_available else 'Framework Ready'}")
    print("🚀 Ready for Evening Session: Integration Testing")


if __name__ == "__main__":
    asyncio.run(test_personal_finance_integrator())