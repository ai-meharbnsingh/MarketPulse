# tests/test_day2_risk_system.py

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "01_Framework_Core"))
sys.path.insert(0, str(project_root))

from src.ai_trading.risk_calculator import KellyCalculator, RiskParameters
from src.ai_trading.ai_risk_assessor import AIRiskAssessor
from antifragile_framework.core.failover_engine import FailoverEngine
from antifragile_framework.config.config_loader import load_provider_profiles
from antifragile_framework.providers.provider_registry import get_default_provider_registry
from telemetry.event_bus import EventBus


async def test_risk_system():
    """Test the complete risk assessment system"""

    print("🧪 Testing Day 2 Risk Assessment System")
    print("=" * 50)

    # Initialize AI framework
    provider_configs = {
        "openai": {"api_keys": ["test"], "resource_config": {}, "circuit_breaker_config": {}},
        "anthropic": {"api_keys": ["test"], "resource_config": {}, "circuit_breaker_config": {}},
        "google_gemini": {"api_keys": ["test"], "resource_config": {}, "circuit_breaker_config": {}}
    }

    try:
        provider_profiles = load_provider_profiles()
        provider_registry = get_default_provider_registry()
        event_bus = EventBus()

        ai_engine = FailoverEngine(
            provider_configs=provider_configs,
            provider_registry=provider_registry,
            event_bus=event_bus,
            provider_profiles=provider_profiles,
        )

        print("✅ AI Framework initialized")

    except Exception as e:
        print(f"❌ AI Framework failed: {e}")
        return False

    # Test Kelly Calculator
    print("\n📊 Testing Kelly Calculator...")

    risk_params = RiskParameters(
        max_daily_loss_percent=2.0,
        max_position_size_percent=5.0,
        max_open_positions=6,
        max_sector_allocation_percent=30.0
    )

    kelly_calc = KellyCalculator(ai_engine, risk_params)

    # Test position size calculation
    kelly_result = kelly_calc.calculate_kelly_position_size(
        win_rate=0.65,  # 65% win rate
        avg_win_percent=8.0,  # 8% average win
        avg_loss_percent=4.0,  # 4% average loss
        current_capital=100000  # 1 lakh capital
    )

    print(f"Kelly Calculator Results:")
    print(f"  Raw Kelly: {kelly_result['raw_kelly_percent']}%")
    print(f"  Conservative Kelly: {kelly_result['conservative_kelly_percent']}%")
    print(f"  Recommended Size: ₹{kelly_result['recommended_position_size']:,.0f}")
    print(f"  Within Limits: {kelly_result['within_limits']}")

    # Test AI Psychology Check
    print("\n🧠 Testing AI Psychology Check...")

    trade_idea = {
        'symbol': 'RELIANCE',
        'reason': 'Strong quarterly results and sector rotation',
        'position_percent': 4.0,
        'recent_pnl': '+5% this week'
    }

    psychology_result = await kelly_calc.ai_psychology_check(trade_idea)
    print(f"Psychology Check Status: {psychology_result['status']}")
    if psychology_result.get('ai_analysis'):
        print(f"AI Analysis Preview: {psychology_result['ai_analysis'][:100]}...")

    # Test Full Risk Assessment
    print("\n🛡️ Testing Full Risk Assessment...")

    risk_assessor = AIRiskAssessor(ai_engine, kelly_calc)

    risk_assessment = await risk_assessor.assess_trade_risk(
        symbol="RELIANCE",
        trade_type="swing_trade",
        entry_price=2500,
        stop_loss=2400,
        target_price=2700,
        position_size_percent=4.0,
        market_data={"trend": "bullish", "volatility": "medium"}
    )

    print(f"Risk Assessment Complete:")
    print(f"  Symbol: {risk_assessment['symbol']}")
    print(f"  Risk-Reward: {risk_assessment['risk_reward_ratio']:.2f}")
    print(f"  AI Recommendation: {risk_assessment['ai_recommendation']}")

    print("\n✅ All risk system tests completed successfully!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_risk_system())
    if success:
        print("\n🎉 Day 2 Risk System is operational!")
    else:
        print("\n❌ Some tests failed - check configuration")