"""
Enterprise Risk Management and Analysis Suite
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall_95: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    tracking_error: float
    downside_deviation: float

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    symbol: str
    position_size: float
    market_value: float
    portfolio_weight: float
    var_contribution: float
    risk_contribution: float
    concentration_score: str  # 'Low', 'Medium', 'High'
    liquidity_score: str
    sector_exposure: str

class RiskManager:
    """Enterprise-grade risk management and analysis"""
    
    # Class-level sets for O(1) lookup (constant-time membership checking)
    LARGE_CAP_STOCKS = frozenset(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'])
    TECH_STOCKS = frozenset(['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA'])
    FINANCE_STOCKS = frozenset(['JPM', 'BAC', 'WFC', 'GS'])
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.risk_limits = {
            'max_position_weight': 0.20,  # Maximum 20% in single position
            'max_sector_weight': 0.30,    # Maximum 30% in single sector
            'max_portfolio_var': 0.05,    # Maximum 5% daily VaR
            'min_liquidity_score': 70,    # Minimum liquidity score
            'max_correlation': 0.80       # Maximum correlation between positions
        }
    
    def calculate_portfolio_risk(self, holdings: List[Dict[str, Any]], 
                               price_history: Dict[str, List[float]]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if not holdings or not price_history:
            return self._empty_risk_metrics()
        
        try:
            # Prepare portfolio data
            portfolio_df = self._prepare_portfolio_data(holdings, price_history)
            
            if portfolio_df.empty:
                return self._empty_risk_metrics()
            
            # Calculate returns
            portfolio_returns = portfolio_df.pct_change().dropna()
            
            if len(portfolio_returns) < 10:  # Need minimum data
                return self._empty_risk_metrics()
            
            # Risk metrics calculations
            volatility = portfolio_returns.std().iloc[0] * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = portfolio_returns.quantile(0.05).iloc[0]
            var_99 = portfolio_returns.quantile(0.01).iloc[0]
            
            # Expected Shortfall (Conditional VaR)
            es_95 = portfolio_returns[portfolio_returns <= var_95].mean().iloc[0]
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min().iloc[0]
            
            # Sharpe Ratio
            mean_return = portfolio_returns.mean().iloc[0] * 252
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino Ratio (downside risk adjusted)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std().iloc[0] * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Calculate beta against market (simplified - use SPY proxy)
            market_returns = self._generate_market_returns(len(portfolio_returns))
            beta = self._calculate_beta(portfolio_returns.iloc[:, 0].values, market_returns)
            
            # Information Ratio and Tracking Error
            excess_returns = portfolio_returns.iloc[:, 0] - (market_returns / 252)  # Daily excess returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            return RiskMetrics(
                value_at_risk_95=var_95,
                value_at_risk_99=var_99,
                expected_shortfall_95=es_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                downside_deviation=downside_deviation
            )
            
        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            return self._empty_risk_metrics()
    
    def _prepare_portfolio_data(self, holdings: List[Dict[str, Any]], 
                              price_history: Dict[str, List[float]]) -> pd.DataFrame:
        """Prepare portfolio data for risk calculations using vectorized operations"""
        try:
            if not price_history:
                return pd.DataFrame()
            
            # Get minimum length across all price histories
            min_length = min(len(prices) for prices in price_history.values())
            
            # Build symbol to shares mapping for efficient lookup
            shares_map = {
                holding.get('symbol', ''): holding.get('shares', 0)
                for holding in holdings
            }
            
            # Vectorized calculation using numpy arrays
            portfolio_values = np.zeros(min_length)
            for symbol, prices in price_history.items():
                shares = shares_map.get(symbol, 0)
                if shares > 0:
                    portfolio_values += shares * np.array(prices[:min_length])
            
            # Create DataFrame
            dates = pd.date_range(start='2023-01-01', periods=min_length, freq='D')
            return pd.DataFrame({'portfolio_value': portfolio_values}, index=dates)
            
        except Exception as e:
            print(f"Error preparing portfolio data: {e}")
            return pd.DataFrame()
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        try:
            if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
                return 1.0  # Default beta
            
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 1.0
        except:
            return 1.0
    
    def _generate_market_returns(self, length: int) -> np.ndarray:
        """Generate synthetic market returns for beta calculation"""
        np.random.seed(42)  # For reproducibility
        return np.random.normal(0.08/252, 0.16/np.sqrt(252), length)  # 8% annual return, 16% volatility
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics for error cases"""
        return RiskMetrics(
            value_at_risk_95=0.0,
            value_at_risk_99=0.0,
            expected_shortfall_95=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            beta=1.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            downside_deviation=0.0
        )
    
    def calculate_position_risks(self, holdings: List[Dict[str, Any]], 
                               total_portfolio_value: float) -> List[PositionRisk]:
        """Calculate risk metrics for individual positions"""
        position_risks = []
        
        for holding in holdings:
            symbol = holding.get('symbol', '')
            shares = holding.get('shares', 0)
            current_price = holding.get('current_price', 0)
            
            market_value = shares * current_price
            portfolio_weight = market_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Simplified risk scoring
            concentration_score = self._get_concentration_score(portfolio_weight)
            liquidity_score = self._get_liquidity_score(symbol)
            sector_exposure = self._get_sector_exposure(symbol)
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=shares,
                market_value=market_value,
                portfolio_weight=portfolio_weight,
                var_contribution=portfolio_weight * 0.02,  # Simplified
                risk_contribution=portfolio_weight,
                concentration_score=concentration_score,
                liquidity_score=liquidity_score,
                sector_exposure=sector_exposure
            )
            
            position_risks.append(position_risk)
        
        return position_risks
    
    def _get_concentration_score(self, weight: float) -> str:
        """Determine concentration risk level"""
        if weight > 0.15:
            return "High"
        elif weight > 0.08:
            return "Medium"
        else:
            return "Low"
    
    def _get_liquidity_score(self, symbol: str) -> str:
        """Simplified liquidity scoring using O(1) set lookup"""
        if symbol in self.LARGE_CAP_STOCKS:
            return "High"
        else:
            return "Medium"
    
    def _get_sector_exposure(self, symbol: str) -> str:
        """Determine sector exposure using O(1) set lookup"""
        if symbol in self.TECH_STOCKS:
            return "Technology"
        elif symbol in self.FINANCE_STOCKS:
            return "Finance"
        else:
            return "Other"
    
    def check_risk_limits(self, position_risks: List[PositionRisk]) -> List[Dict[str, Any]]:
        """Check for risk limit violations"""
        violations = []
        
        for position in position_risks:
            # Check position concentration
            if position.portfolio_weight > self.risk_limits['max_position_weight']:
                violations.append({
                    'type': 'position_concentration',
                    'symbol': position.symbol,
                    'current_weight': position.portfolio_weight,
                    'limit': self.risk_limits['max_position_weight'],
                    'severity': 'High'
                })
        
        return violations
