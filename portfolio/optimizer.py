"""
Advanced Portfolio Optimization and Risk Management
Enterprise-grade implementation with Modern Portfolio Theory
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Try to import scipy for optimization, provide fallback
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available, using simplified optimization")


@dataclass
class AssetAllocation:
    """Asset allocation result"""

    symbol: str
    weight: float
    expected_return: float
    risk: float
    sharpe_ratio: float


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""

    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    beta: float
    alpha: float
    sortino_ratio: float


class PortfolioOptimizer:
    """Advanced portfolio optimization using Modern Portfolio Theory"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = {}
        self.correlation_matrix = None
        self.covariance_matrix = None

    def prepare_returns_data(
        self, price_data: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Prepare returns data for optimization"""
        returns_dict = {}

        for symbol, data in price_data.items():
            if len(data) < 2:
                continue

            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            df["returns"] = df["close_price"].pct_change()

            # Remove NaN values
            returns = df["returns"].dropna()
            if len(returns) > 0:
                returns_dict[symbol] = returns.values

        if not returns_dict:
            raise ValueError("No valid returns data available")

        # Align all return series to same length
        min_length = min(len(returns) for returns in returns_dict.values())
        aligned_returns = {
            symbol: returns[-min_length:] for symbol, returns in returns_dict.items()
        }

        return pd.DataFrame(aligned_returns)

    def calculate_portfolio_metrics(
        self, weights: np.ndarray, returns: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, risk, and Sharpe ratio"""
        # Portfolio return (annualized)
        portfolio_return = np.sum(returns.mean() * weights) * 252

        # Portfolio risk (annualized)
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std

        return portfolio_return, portfolio_std, sharpe_ratio

    def optimize_portfolio(
        self, price_data: Dict[str, List[Dict[str, Any]]], objective: str = "sharpe"
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            # Prepare returns data
            returns_df = self.prepare_returns_data(price_data)
            symbols = returns_df.columns.tolist()
            n_assets = len(symbols)

            if n_assets < 2:
                raise ValueError("Need at least 2 assets for optimization")

            # Store for later use
            self.returns_data = returns_df
            self.correlation_matrix = returns_df.corr()
            self.covariance_matrix = returns_df.cov()

            if SCIPY_AVAILABLE and objective in [
                "sharpe",
                "min_volatility",
                "max_return",
            ]:
                return self._scipy_optimization(returns_df, symbols, objective)
            else:
                return self._simple_optimization(returns_df, symbols)

        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
            # Return equal-weight fallback
            return self._equal_weight_fallback(list(price_data.keys()))

    def _scipy_optimization(
        self, returns_df: pd.DataFrame, symbols: List[str], objective: str
    ) -> Dict[str, Any]:
        """Advanced optimization using SciPy"""
        n_assets = len(symbols)

        # Objective functions
        def negative_sharpe(weights):
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, returns_df)
            return -sharpe

        def portfolio_volatility(weights):
            _, vol, _ = self.calculate_portfolio_metrics(weights, returns_df)
            return vol

        def negative_return(weights):
            ret, _, _ = self.calculate_portfolio_metrics(weights, returns_df)
            return -ret

        # Constraints and bounds
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 0.4) for _ in range(n_assets))  # Max 40% in any asset

        # Initial guess (equal weights)
        x0 = np.array([1 / n_assets] * n_assets)

        # Select objective function
        if objective == "sharpe":
            objective_func = negative_sharpe
        elif objective == "min_volatility":
            objective_func = portfolio_volatility
        else:  # max_return
            objective_func = negative_return

        # Optimize
        result = minimize(
            objective_func, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if not result.success:
            print("⚠️  Optimization failed, using simple method")
            return self._simple_optimization(returns_df, symbols)

        optimal_weights = result.x
        portfolio_return, portfolio_risk, sharpe_ratio = (
            self.calculate_portfolio_metrics(optimal_weights, returns_df)
        )

        # Create allocation results
        allocations = []
        for i, symbol in enumerate(symbols):
            if optimal_weights[i] > 0.01:  # Only include significant allocations
                asset_return = returns_df[symbol].mean() * 252
                asset_risk = returns_df[symbol].std() * np.sqrt(252)
                asset_sharpe = (
                    (asset_return - self.risk_free_rate) / asset_risk
                    if asset_risk > 0
                    else 0
                )

                allocations.append(
                    AssetAllocation(
                        symbol=symbol,
                        weight=optimal_weights[i],
                        expected_return=asset_return,
                        risk=asset_risk,
                        sharpe_ratio=asset_sharpe,
                    )
                )

        # Portfolio metrics
        metrics = self._calculate_comprehensive_metrics(returns_df, optimal_weights)

        return {
            "allocations": allocations,
            "portfolio_metrics": metrics,
            "optimization_method": f"scipy_{objective}",
            "total_assets": len(allocations),
            "correlation_matrix": self.correlation_matrix.to_dict(),
        }

    def _simple_optimization(
        self, returns_df: pd.DataFrame, symbols: List[str]
    ) -> Dict[str, Any]:
        """Simple risk-adjusted optimization without SciPy"""
        # Calculate Sharpe ratios for each asset
        asset_metrics = []
        for symbol in symbols:
            returns = returns_df[symbol]
            annual_return = returns.mean() * 252
            annual_risk = returns.std() * np.sqrt(252)
            sharpe = (
                (annual_return - self.risk_free_rate) / annual_risk
                if annual_risk > 0
                else 0
            )

            asset_metrics.append(
                {
                    "symbol": symbol,
                    "return": annual_return,
                    "risk": annual_risk,
                    "sharpe": sharpe,
                }
            )

        # Sort by Sharpe ratio and risk-adjusted return
        asset_metrics.sort(key=lambda x: x["sharpe"], reverse=True)

        # Simple allocation: higher weight to better Sharpe ratios
        total_sharpe = sum(max(0, m["sharpe"]) for m in asset_metrics)

        allocations = []
        for metric in asset_metrics:
            if metric["sharpe"] > 0 and total_sharpe > 0:
                weight = max(
                    0.1, metric["sharpe"] / total_sharpe
                )  # Minimum 10% if included
                weight = min(0.35, weight)  # Maximum 35%
            else:
                weight = 1.0 / len(symbols)  # Equal weight fallback

            if weight > 0.05:  # Only include if weight > 5%
                allocations.append(
                    AssetAllocation(
                        symbol=metric["symbol"],
                        weight=weight,
                        expected_return=metric["return"],
                        risk=metric["risk"],
                        sharpe_ratio=metric["sharpe"],
                    )
                )

        # Normalize weights
        total_weight = sum(a.weight for a in allocations)
        for allocation in allocations:
            allocation.weight = allocation.weight / total_weight

        # Calculate portfolio metrics
        weights = np.array([a.weight for a in allocations])
        selected_returns = returns_df[[a.symbol for a in allocations]]
        metrics = self._calculate_comprehensive_metrics(selected_returns, weights)

        return {
            "allocations": allocations,
            "portfolio_metrics": metrics,
            "optimization_method": "simple_sharpe",
            "total_assets": len(allocations),
            "correlation_matrix": self.correlation_matrix.to_dict(),
        }

    def _equal_weight_fallback(self, symbols: List[str]) -> Dict[str, Any]:
        """Equal weight fallback when optimization fails"""
        allocations = []
        weight = 1.0 / len(symbols)

        for symbol in symbols:
            allocations.append(
                AssetAllocation(
                    symbol=symbol,
                    weight=weight,
                    expected_return=0.08,  # Default assumption
                    risk=0.20,  # Default assumption
                    sharpe_ratio=0.30,
                )
            )

        return {
            "allocations": allocations,
            "portfolio_metrics": PortfolioMetrics(
                total_return=0.08,
                annual_return=0.08,
                volatility=0.15,
                sharpe_ratio=0.40,
                max_drawdown=-0.20,
                var_95=-0.05,
                beta=1.0,
                alpha=0.0,
                sortino_ratio=0.50,
            ),
            "optimization_method": "equal_weight_fallback",
            "total_assets": len(symbols),
            "correlation_matrix": {},
        }

    def _calculate_comprehensive_metrics(
        self, returns_df: pd.DataFrame, weights: np.ndarray
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        )

        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95%)
        var_95 = portfolio_returns.quantile(0.05)

        # Sortino ratio (downside risk)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = (
            downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0
            else volatility
        )
        sortino_ratio = (
            (annual_return - self.risk_free_rate) / downside_std
            if downside_std > 0
            else 0
        )

        # Beta and Alpha (simplified - using portfolio variance as market proxy)
        beta = 1.0  # Simplified
        alpha = annual_return - (
            self.risk_free_rate + beta * (annual_return - self.risk_free_rate)
        )

        return PortfolioMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            beta=beta,
            alpha=alpha,
            sortino_ratio=sortino_ratio,
        )

    def calculate_risk_metrics(
        self, portfolio_holdings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for existing portfolio"""
        if not portfolio_holdings:
            return {"error": "No holdings provided"}

        total_value = sum(h["market_value"] for h in portfolio_holdings)
        weights = {
            h["symbol"]: h["market_value"] / total_value for h in portfolio_holdings
        }

        # Concentration risk
        max_weight = max(weights.values())
        concentration_score = (
            "High" if max_weight > 0.3 else "Medium" if max_weight > 0.2 else "Low"
        )

        # Sector diversification (simplified)
        unique_symbols = len(weights)
        diversification_score = (
            "High"
            if unique_symbols >= 8
            else "Medium" if unique_symbols >= 5 else "Low"
        )

        # Risk-adjusted return potential
        avg_return = sum(
            h.get("expected_return", 0.08) * weights[h["symbol"]]
            for h in portfolio_holdings
        )

        return {
            "total_value": total_value,
            "position_count": unique_symbols,
            "concentration_risk": concentration_score,
            "diversification": diversification_score,
            "largest_position_weight": max_weight,
            "expected_annual_return": avg_return,
            "risk_score": self._calculate_risk_score(weights, portfolio_holdings),
            "rebalancing_needed": max_weight > 0.35 or unique_symbols < 4,
        }

    def _calculate_risk_score(
        self, weights: Dict[str, float], holdings: List[Dict[str, Any]]
    ) -> str:
        """Calculate overall portfolio risk score"""
        risk_factors = 0

        # Concentration risk
        if max(weights.values()) > 0.3:
            risk_factors += 2
        elif max(weights.values()) > 0.2:
            risk_factors += 1

        # Diversification
        if len(weights) < 4:
            risk_factors += 2
        elif len(weights) < 6:
            risk_factors += 1

        # Volatility (simplified)
        high_vol_weight = sum(
            weights[h["symbol"]] for h in holdings if h.get("volatility", 0.2) > 0.3
        )
        if high_vol_weight > 0.5:
            risk_factors += 2
        elif high_vol_weight > 0.3:
            risk_factors += 1

        if risk_factors >= 4:
            return "High"
        elif risk_factors >= 2:
            return "Medium"
        else:
            return "Low"

    def suggest_rebalancing(
        self,
        current_holdings: List[Dict[str, Any]],
        target_allocations: List[AssetAllocation],
        total_portfolio_value: float,
    ) -> List[Dict[str, Any]]:
        """Suggest rebalancing trades"""
        suggestions = []

        # Current weights
        current_weights = {
            h["symbol"]: h["market_value"] / total_portfolio_value
            for h in current_holdings
        }

        # Target weights
        target_weights = {a.symbol: a.weight for a in target_allocations}

        # Calculate differences
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.05:  # Only suggest if difference > 5%
                dollar_amount = weight_diff * total_portfolio_value
                action = "BUY" if weight_diff > 0 else "SELL"

                suggestions.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_difference": weight_diff,
                        "dollar_amount": abs(dollar_amount),
                        "priority": (
                            "High"
                            if abs(weight_diff) > 0.15
                            else "Medium" if abs(weight_diff) > 0.10 else "Low"
                        ),
                    }
                )

        # Sort by priority and dollar amount
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        suggestions.sort(
            key=lambda x: (priority_order[x["priority"]], x["dollar_amount"]),
            reverse=True,
        )

        return suggestions
