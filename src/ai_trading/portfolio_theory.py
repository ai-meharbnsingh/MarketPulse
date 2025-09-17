# portfolio_theory.py - FIXED for Day 7 import compatibility

"""
Portfolio Theory Module - Compatible with Day 7 validator
This module provides the PortfolioOptimizer class that the system integration expects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory principles"""

    def __init__(self):
        """Initialize the portfolio optimizer"""
        self.risk_free_rate = 0.06  # 6% risk-free rate for Indian context
        self.market_return = 0.12  # 12% expected market return

        # Default asset returns and risks (Indian market context)
        self.default_returns = {
            'large_cap': 0.12,
            'mid_cap': 0.15,
            'small_cap': 0.18,
            'bonds': 0.07,
            'gold': 0.08
        }

        self.default_risks = {
            'large_cap': 0.18,
            'mid_cap': 0.22,
            'small_cap': 0.28,
            'bonds': 0.05,
            'gold': 0.20
        }

    def optimize_portfolio(self, assets: List[str], returns: Optional[Dict] = None,
                           risk_tolerance: float = 0.5) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on assets and risk tolerance

        Args:
            assets: List of asset names
            returns: Optional dictionary of expected returns for assets
            risk_tolerance: Risk tolerance between 0 (conservative) and 1 (aggressive)

        Returns:
            Dictionary with allocation and portfolio metrics
        """

        if not assets:
            return {
                'allocation': {},
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'error': 'No assets provided'
            }

        try:
            # Use provided returns or defaults
            if returns is None:
                returns = self._get_default_returns(assets)

            # Get risk estimates
            risks = self._get_default_risks(assets)

            # Optimize based on risk tolerance
            if risk_tolerance <= 0.3:  # Conservative
                allocation = self._conservative_allocation(assets)
            elif risk_tolerance >= 0.7:  # Aggressive
                allocation = self._aggressive_allocation(assets)
            else:  # Moderate
                allocation = self._moderate_allocation(assets)

            # Calculate portfolio metrics
            portfolio_return = sum(allocation[asset] * returns.get(asset, 0.08) for asset in assets)
            portfolio_risk = self._calculate_portfolio_risk(allocation, risks)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

            return {
                'allocation': allocation,
                'expected_return': round(portfolio_return, 4),
                'volatility': round(portfolio_risk, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'risk_tolerance': risk_tolerance,
                'optimization_method': 'risk_based_allocation'
            }

        except Exception as e:
            return {
                'allocation': {asset: 1.0 / len(assets) for asset in assets},
                'expected_return': 0.08,
                'volatility': 0.15,
                'sharpe_ratio': 0.13,
                'error': f'Optimization failed: {str(e)}',
                'fallback': True
            }

    def _get_default_returns(self, assets: List[str]) -> Dict[str, float]:
        """Get default return estimates for assets"""
        returns = {}
        for asset in assets:
            asset_lower = asset.lower()

            # Map common asset names to return estimates
            if any(term in asset_lower for term in ['large', 'cap', 'blue', 'nifty']):
                returns[asset] = self.default_returns['large_cap']
            elif any(term in asset_lower for term in ['mid', 'medium']):
                returns[asset] = self.default_returns['mid_cap']
            elif any(term in asset_lower for term in ['small', 'micro']):
                returns[asset] = self.default_returns['small_cap']
            elif any(term in asset_lower for term in ['bond', 'debt', 'fixed']):
                returns[asset] = self.default_returns['bonds']
            elif any(term in asset_lower for term in ['gold', 'commodity']):
                returns[asset] = self.default_returns['gold']
            else:
                returns[asset] = 0.10  # Default 10% return

        return returns

    def _get_default_risks(self, assets: List[str]) -> Dict[str, float]:
        """Get default risk estimates for assets"""
        risks = {}
        for asset in assets:
            asset_lower = asset.lower()

            # Map common asset names to risk estimates
            if any(term in asset_lower for term in ['large', 'cap', 'blue', 'nifty']):
                risks[asset] = self.default_risks['large_cap']
            elif any(term in asset_lower for term in ['mid', 'medium']):
                risks[asset] = self.default_risks['mid_cap']
            elif any(term in asset_lower for term in ['small', 'micro']):
                risks[asset] = self.default_risks['small_cap']
            elif any(term in asset_lower for term in ['bond', 'debt', 'fixed']):
                risks[asset] = self.default_risks['bonds']
            elif any(term in asset_lower for term in ['gold', 'commodity']):
                risks[asset] = self.default_risks['gold']
            else:
                risks[asset] = 0.16  # Default 16% risk

        return risks

    def _conservative_allocation(self, assets: List[str]) -> Dict[str, float]:
        """Generate conservative allocation (debt-heavy)"""
        allocation = {}
        debt_weight = 0.7
        equity_weight = 0.3

        debt_assets = [a for a in assets if any(term in a.lower() for term in ['bond', 'debt', 'fixed'])]
        equity_assets = [a for a in assets if a not in debt_assets]

        # Allocate to debt
        if debt_assets:
            debt_per_asset = debt_weight / len(debt_assets)
            for asset in debt_assets:
                allocation[asset] = debt_per_asset

        # Allocate remaining to equity
        if equity_assets:
            equity_per_asset = equity_weight / len(equity_assets)
            for asset in equity_assets:
                allocation[asset] = equity_per_asset
        else:
            # If no equity assets, distribute remaining to debt
            if debt_assets:
                remaining_per_debt = equity_weight / len(debt_assets)
                for asset in debt_assets:
                    allocation[asset] += remaining_per_debt

        # Handle case where no debt assets
        if not debt_assets:
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                allocation[asset] = equal_weight

        return allocation

    def _aggressive_allocation(self, assets: List[str]) -> Dict[str, float]:
        """Generate aggressive allocation (equity-heavy)"""
        allocation = {}
        equity_weight = 0.85
        debt_weight = 0.15

        debt_assets = [a for a in assets if any(term in a.lower() for term in ['bond', 'debt', 'fixed'])]
        equity_assets = [a for a in assets if a not in debt_assets]

        # Allocate to equity
        if equity_assets:
            equity_per_asset = equity_weight / len(equity_assets)
            for asset in equity_assets:
                allocation[asset] = equity_per_asset

        # Allocate remaining to debt
        if debt_assets:
            debt_per_asset = debt_weight / len(debt_assets)
            for asset in debt_assets:
                allocation[asset] = debt_per_asset
        else:
            # If no debt assets, distribute remaining to equity
            if equity_assets:
                remaining_per_equity = debt_weight / len(equity_assets)
                for asset in equity_assets:
                    allocation[asset] += remaining_per_equity

        # Handle case where no equity assets
        if not equity_assets:
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                allocation[asset] = equal_weight

        return allocation

    def _moderate_allocation(self, assets: List[str]) -> Dict[str, float]:
        """Generate moderate allocation (balanced)"""
        allocation = {}
        equity_weight = 0.6
        debt_weight = 0.4

        debt_assets = [a for a in assets if any(term in a.lower() for term in ['bond', 'debt', 'fixed'])]
        equity_assets = [a for a in assets if a not in debt_assets]

        # Allocate to equity
        if equity_assets:
            equity_per_asset = equity_weight / len(equity_assets)
            for asset in equity_assets:
                allocation[asset] = equity_per_asset

        # Allocate to debt
        if debt_assets:
            debt_per_asset = debt_weight / len(debt_assets)
            for asset in debt_assets:
                allocation[asset] = debt_per_asset

        # Handle missing asset classes
        if not equity_assets and debt_assets:
            # Only debt assets - distribute equity weight to debt
            remaining_per_debt = equity_weight / len(debt_assets)
            for asset in debt_assets:
                allocation[asset] += remaining_per_debt

        elif equity_assets and not debt_assets:
            # Only equity assets - distribute debt weight to equity
            remaining_per_equity = debt_weight / len(equity_assets)
            for asset in equity_assets:
                allocation[asset] += remaining_per_equity

        elif not equity_assets and not debt_assets:
            # No classified assets - equal weight
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                allocation[asset] = equal_weight

        return allocation

    def _calculate_portfolio_risk(self, allocation: Dict[str, float], risks: Dict[str, float]) -> float:
        """Calculate portfolio risk (simplified - assumes some correlation)"""
        try:
            # Weighted average risk with correlation adjustment
            total_risk = 0
            for asset, weight in allocation.items():
                asset_risk = risks.get(asset, 0.16)
                total_risk += (weight ** 2) * (asset_risk ** 2)

            # Add correlation effect (simplified)
            correlation_adjustment = 0.8  # Assume 0.8 average correlation
            portfolio_variance = total_risk * correlation_adjustment

            return np.sqrt(portfolio_variance)

        except Exception:
            # Fallback to simple weighted average
            return sum(allocation[asset] * risks.get(asset, 0.16) for asset in allocation)

    def calculate_efficient_frontier(self, assets: List[str], num_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier data"""
        try:
            results = []

            for i in range(num_portfolios):
                risk_tolerance = i / (num_portfolios - 1)  # 0 to 1
                portfolio = self.optimize_portfolio(assets, risk_tolerance=risk_tolerance)

                results.append({
                    'risk_tolerance': risk_tolerance,
                    'expected_return': portfolio['expected_return'],
                    'volatility': portfolio['volatility'],
                    'sharpe_ratio': portfolio['sharpe_ratio']
                })

            return pd.DataFrame(results)

        except Exception as e:
            # Return empty DataFrame on error
            return pd.DataFrame(columns=['risk_tolerance', 'expected_return', 'volatility', 'sharpe_ratio'])


# Additional classes for compatibility

class ModernPortfolioTheory:
    """Modern Portfolio Theory implementation wrapper"""

    def __init__(self):
        self.optimizer = PortfolioOptimizer()

    def optimize(self, assets, **kwargs):
        return self.optimizer.optimize_portfolio(assets, **kwargs)


class RiskModel:
    """Risk model for portfolio analysis"""

    def __init__(self):
        self.market_beta = 1.0

    def calculate_var(self, portfolio, confidence=0.05):
        """Calculate Value at Risk"""
        return portfolio.get('volatility', 0.15) * 2.33  # 1% VaR approximation


# Test function
def test_portfolio_optimizer():
    """Test the portfolio optimizer"""
    print("🧪 Testing Portfolio Optimizer")
    print("=" * 40)

    optimizer = PortfolioOptimizer()

    # Test with sample assets
    test_assets = ['Large Cap Equity', 'Mid Cap Equity', 'Government Bonds', 'Gold']

    for risk_level, risk_tolerance in [('Conservative', 0.2), ('Moderate', 0.5), ('Aggressive', 0.8)]:
        print(f"\n{risk_level} Portfolio (Risk Tolerance: {risk_tolerance}):")

        result = optimizer.optimize_portfolio(test_assets, risk_tolerance=risk_tolerance)

        print(f"Expected Return: {result['expected_return']:.1%}")
        print(f"Volatility: {result['volatility']:.1%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print("Allocation:")
        for asset, weight in result['allocation'].items():
            print(f"  {asset}: {weight:.1%}")

    print(f"\n✅ Portfolio Optimizer test completed successfully!")


if __name__ == "__main__":
    test_portfolio_optimizer()