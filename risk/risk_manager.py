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
            downside_deviation = downside_returns.std().iloc[0] * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = abs(mean_return / max_drawdown) if max_drawdown < 0 else 0
            
            # Beta (simplified - using portfolio variance as market proxy)
            beta = 1.0  # Simplified for now
            
            # Information Ratio and Tracking Error (simplified)
            tracking_error = volatility * 0.5  # Simplified
            information_ratio = (mean_return - self.risk_free_rate) / tracking_error if tracking_error > 0 else 0
            
            return RiskMetrics(
                value_at_risk_95=abs(var_95),
                value_at_risk_99=abs(var_99),
                expected_shortfall_95=abs(es_95),
                max_drawdown=abs(max_drawdown),
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
            print(f"❌ Risk calculation error: {e}")
            return self._empty_risk_metrics()
    
    def analyze_position_risk(self, holdings: List[Dict[str, Any]], 
                            total_portfolio_value: float) -> List[PositionRisk]:
        """Analyze risk for individual positions"""
        position_risks = []
        
        for holding in holdings:
            symbol = holding['symbol']
            market_value = holding.get('market_value', 0)
            portfolio_weight = market_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Concentration risk
            concentration_score = self._assess_concentration_risk(portfolio_weight)
            
            # Liquidity risk (simplified)
            liquidity_score = self._assess_liquidity_risk(symbol, market_value)
            
            # Sector exposure (simplified)
            sector = self._get_sector(symbol)
            
            # VaR contribution (simplified)
            var_contribution = portfolio_weight * 0.02  # Simplified
            risk_contribution = var_contribution
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=holding.get('shares', 0),
                market_value=market_value,
                portfolio_weight=portfolio_weight,
                var_contribution=var_contribution,
                risk_contribution=risk_contribution,
                concentration_score=concentration_score,
                liquidity_score=liquidity_score,
                sector_exposure=sector
            )
            
            position_risks.append(position_risk)
        
        return position_risks
    
    def check_risk_limits(self, holdings: List[Dict[str, Any]], 
                         portfolio_metrics: RiskMetrics) -> Dict[str, Any]:
        """Check portfolio against risk limits"""
        violations = []
        warnings = []
        
        total_value = sum(h.get('market_value', 0) for h in holdings)
        
        # Position concentration checks
        for holding in holdings:
            weight = holding.get('market_value', 0) / total_value if total_value > 0 else 0
            
            if weight > self.risk_limits['max_position_weight']:
                violations.append({
                    'type': 'Position Concentration',
                    'symbol': holding['symbol'],
                    'current': f"{weight:.1%}",
                    'limit': f"{self.risk_limits['max_position_weight']:.1%}",
                    'severity': 'HIGH'
                })
            elif weight > self.risk_limits['max_position_weight'] * 0.8:
                warnings.append({
                    'type': 'Position Concentration',
                    'symbol': holding['symbol'],
                    'current': f"{weight:.1%}",
                    'limit': f"{self.risk_limits['max_position_weight']:.1%}",
                    'severity': 'MEDIUM'
                })
        
        # Portfolio VaR check
        if portfolio_metrics.value_at_risk_95 > self.risk_limits['max_portfolio_var']:
            violations.append({
                'type': 'Portfolio VaR',
                'current': f"{portfolio_metrics.value_at_risk_95:.2%}",
                'limit': f"{self.risk_limits['max_portfolio_var']:.2%}",
                'severity': 'HIGH'
            })
        
        # Sector concentration (simplified)
        sector_weights = self._calculate_sector_weights(holdings, total_value)
        for sector, weight in sector_weights.items():
            if weight > self.risk_limits['max_sector_weight']:
                violations.append({
                    'type': 'Sector Concentration',
                    'sector': sector,
                    'current': f"{weight:.1%}",
                    'limit': f"{self.risk_limits['max_sector_weight']:.1%}",
                    'severity': 'MEDIUM'
                })
        
        return {
            'violations': violations,
            'warnings': warnings,
            'compliance_score': self._calculate_compliance_score(violations, warnings),
            'risk_level': self._assess_overall_risk_level(violations, warnings, portfolio_metrics)
        }
    
    def calculate_correlation_matrix(self, price_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate correlation matrix for portfolio holdings"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_history)
            
            if df.empty or len(df.columns) < 2:
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Correlation matrix
            correlation_matrix = returns.corr()
            
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > self.risk_limits['max_correlation']:
                        high_correlations.append({
                            'asset1': correlation_matrix.columns[i],
                            'asset2': correlation_matrix.columns[j],
                            'correlation': round(corr_value, 3)
                        })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': high_correlations,
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
            }
            
        except Exception as e:
            return {'error': f'Correlation calculation failed: {e}'}
    
    def stress_test_portfolio(self, holdings: List[Dict[str, Any]], 
                            scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        stress_results = []
        
        default_scenarios = [
            {'name': 'Market Crash (-20%)', 'market_shock': -0.20, 'volatility_shock': 2.0},
            {'name': 'Economic Recession (-15%)', 'market_shock': -0.15, 'volatility_shock': 1.5},
            {'name': 'Interest Rate Shock (+5%)', 'market_shock': -0.10, 'volatility_shock': 1.3},
            {'name': 'Sector Rotation (-10%)', 'market_shock': -0.10, 'volatility_shock': 1.0}
        ]
        
        scenarios_to_test = scenarios if scenarios else default_scenarios
        
        total_value = sum(h.get('market_value', 0) for h in holdings)
        
        for scenario in scenarios_to_test:
            market_shock = scenario.get('market_shock', -0.10)
            volatility_shock = scenario.get('volatility_shock', 1.5)
            
            # Calculate portfolio impact
            portfolio_impact = total_value * market_shock
            new_portfolio_value = total_value + portfolio_impact
            
            # Adjust VaR for increased volatility
            base_var = total_value * 0.02  # Assume 2% base VaR
            stressed_var = base_var * volatility_shock
            
            stress_results.append({
                'scenario': scenario.get('name', 'Unnamed Scenario'),
                'portfolio_loss': abs(portfolio_impact),
                'portfolio_loss_percent': abs(market_shock),
                'new_portfolio_value': new_portfolio_value,
                'stressed_var': stressed_var,
                'recovery_time_estimate': abs(market_shock) * 12  # Simplified: months to recover
            })
        
        return {
            'stress_test_results': stress_results,
            'worst_case_loss': max(r['portfolio_loss'] for r in stress_results),
            'worst_case_scenario': max(stress_results, key=lambda x: x['portfolio_loss'])['scenario'],
            'average_loss': sum(r['portfolio_loss'] for r in stress_results) / len(stress_results),
            'risk_capacity_used': max(r['portfolio_loss_percent'] for r in stress_results)
        }
    
    def generate_risk_report(self, holdings: List[Dict[str, Any]], 
                           price_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        total_value = sum(h.get('market_value', 0) for h in holdings)
        
        # Core risk metrics
        portfolio_risk = self.calculate_portfolio_risk(holdings, price_history)
        position_risks = self.analyze_position_risk(holdings, total_value)
        compliance_check = self.check_risk_limits(holdings, portfolio_risk)
        correlation_analysis = self.calculate_correlation_matrix(price_history)
        stress_test = self.stress_test_portfolio(holdings, [])
        
        return {
            'report_date': datetime.now().isoformat(),
            'portfolio_value': total_value,
            'position_count': len(holdings),
            'risk_metrics': {
                'var_95': portfolio_risk.value_at_risk_95,
                'var_99': portfolio_risk.value_at_risk_99,
                'max_drawdown': portfolio_risk.max_drawdown,
                'volatility': portfolio_risk.volatility,
                'sharpe_ratio': portfolio_risk.sharpe_ratio,
                'sortino_ratio': portfolio_risk.sortino_ratio
            },
            'position_risks': [
                {
                    'symbol': pr.symbol,
                    'portfolio_weight': pr.portfolio_weight,
                    'concentration_score': pr.concentration_score,
                    'liquidity_score': pr.liquidity_score,
                    'sector': pr.sector_exposure
                } for pr in position_risks
            ],
            'compliance': compliance_check,
            'correlation_analysis': correlation_analysis,
            'stress_testing': stress_test,
            'recommendations': self._generate_risk_recommendations(
                portfolio_risk, position_risks, compliance_check
            )
        }
    
    def _prepare_portfolio_data(self, holdings: List[Dict[str, Any]], 
                              price_history: Dict[str, List[float]]) -> pd.DataFrame:
        """Prepare portfolio data for analysis"""
        portfolio_data = {}
        total_value = sum(h.get('market_value', 0) for h in holdings)
        
        for holding in holdings:
            symbol = holding['symbol']
            weight = holding.get('market_value', 0) / total_value if total_value > 0 else 0
            
            if symbol in price_history and len(price_history[symbol]) > 0:
                prices = price_history[symbol]
                # Weight the prices by portfolio allocation
                portfolio_data[symbol] = [p * weight for p in prices]
        
        if not portfolio_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(portfolio_data)
        # Sum across all positions to get portfolio value
        df['portfolio'] = df.sum(axis=1)
        return df[['portfolio']]
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            value_at_risk_95=0.0,
            value_at_risk_99=0.0,
            expected_shortfall_95=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            beta=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            downside_deviation=0.0
        )
    
    def _assess_concentration_risk(self, weight: float) -> str:
        """Assess concentration risk for position"""
        if weight > 0.20:
            return "High"
        elif weight > 0.10:
            return "Medium"
        else:
            return "Low"
    
    def _assess_liquidity_risk(self, symbol: str, market_value: float) -> str:
        """Assess liquidity risk (simplified)"""
        # Simplified liquidity scoring
        large_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        if symbol in large_caps:
            return "High"
        elif market_value > 1000000:  # Large position
            return "Medium"
        else:
            return "Low"
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified)"""
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy'
        }
        return sectors.get(symbol, 'Other')
    
    def _calculate_sector_weights(self, holdings: List[Dict[str, Any]], 
                                total_value: float) -> Dict[str, float]:
        """Calculate sector weights"""
        sector_weights = {}
        
        for holding in holdings:
            sector = self._get_sector(holding['symbol'])
            weight = holding.get('market_value', 0) / total_value if total_value > 0 else 0
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights
    
    def _calculate_compliance_score(self, violations: List, warnings: List) -> float:
        """Calculate compliance score (0-100)"""
        base_score = 100
        violation_penalty = len(violations) * 15
        warning_penalty = len(warnings) * 5
        
        return max(0, base_score - violation_penalty - warning_penalty)
    
    def _assess_overall_risk_level(self, violations: List, warnings: List, 
                                 metrics: RiskMetrics) -> str:
        """Assess overall portfolio risk level"""
        if len(violations) > 0 or metrics.volatility > 0.25:
            return "HIGH"
        elif len(warnings) > 0 or metrics.volatility > 0.15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_risk_recommendations(self, portfolio_risk: RiskMetrics, 
                                     position_risks: List[PositionRisk],
                                     compliance: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # High concentration positions
        high_concentration = [pr for pr in position_risks if pr.concentration_score == "High"]
        if high_concentration:
            symbols = [pr.symbol for pr in high_concentration]
            recommendations.append(f"Consider reducing concentration in: {', '.join(symbols)}")
        
        # High volatility
        if portfolio_risk.volatility > 0.20:
            recommendations.append("Portfolio volatility is elevated. Consider diversifying or reducing position sizes.")
        
        # Poor risk-adjusted returns
        if portfolio_risk.sharpe_ratio < 0.5:
            recommendations.append("Sharpe ratio is below optimal. Consider rebalancing toward higher-return assets.")
        
        # Compliance violations
        if compliance['violations']:
            recommendations.append("Address risk limit violations immediately to maintain compliance.")
        
        # Low diversification
        if len(position_risks) < 10:
            recommendations.append("Consider adding more positions to improve diversification.")
        
        return recommendations if recommendations else ["Portfolio risk profile is within acceptable parameters."]