"""
Unit tests for technical indicators
"""
import pytest
import numpy as np
from indicators.technical_indicators import TechnicalIndicators, IndicatorAnalyzer

class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators"""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data"""
        np.random.seed(42)
        base_price = 100
        prices = [base_price]
        for _ in range(100):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(1.0, new_price))
        return prices[1:]  # Remove first element
    
    @pytest.fixture
    def sample_ohlc(self):
        """Generate sample OHLC data"""
        np.random.seed(42)
        data = {'high': [], 'low': [], 'close': []}
        close = 100
        
        for _ in range(50):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            close = close * (1 + np.random.normal(0, 0.02))
            close = max(1.0, close)
            
            data['high'].append(high)
            data['low'].append(low)
            data['close'].append(close)
        
        return data
    
    def test_sma_calculation(self, sample_prices):
        """Test Simple Moving Average calculation"""
        period = 10
        sma = TechnicalIndicators.sma(sample_prices, period)
        
        assert len(sma) == len(sample_prices) - period + 1
        assert all(isinstance(val, float) for val in sma)
        
        # Check first SMA value manually
        expected_first = sum(sample_prices[:period]) / period
        assert abs(sma[0] - expected_first) < 0.0001
    
    def test_ema_calculation(self, sample_prices):
        """Test Exponential Moving Average calculation"""
        period = 10
        ema = TechnicalIndicators.ema(sample_prices, period)
        
        assert len(ema) > 0
        assert all(isinstance(val, float) for val in ema)
        assert len(ema) == len(sample_prices) - period + 1
    
    def test_rsi_calculation(self, sample_prices):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.rsi(sample_prices, 14)
        
        assert len(rsi) > 0
        assert all(0 <= val <= 100 for val in rsi)
        assert all(isinstance(val, float) for val in rsi)
    
    def test_macd_calculation(self, sample_prices):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(sample_prices)
        
        assert len(macd_line) > 0
        assert len(signal_line) > 0
        assert len(histogram) > 0
        assert len(histogram) <= len(macd_line)
    
    def test_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_prices, 20, 2)
        
        assert len(upper) == len(middle) == len(lower)
        assert all(u >= m >= l for u, m, l in zip(upper, middle, lower))
    
    def test_atr_calculation(self, sample_ohlc):
        """Test Average True Range calculation"""
        atr = TechnicalIndicators.atr(
            sample_ohlc['high'],
            sample_ohlc['low'],
            sample_ohlc['close'],
            14
        )
        
        assert len(atr) > 0
        assert all(val >= 0 for val in atr)
    
    def test_stochastic_oscillator(self, sample_ohlc):
        """Test Stochastic Oscillator calculation"""
        k_values, d_values = TechnicalIndicators.stochastic_oscillator(
            sample_ohlc['high'],
            sample_ohlc['low'],
            sample_ohlc['close']
        )
        
        assert len(k_values) > 0
        assert len(d_values) > 0
        assert all(0 <= val <= 100 for val in k_values)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_prices = [100, 101, 102]
        
        sma = TechnicalIndicators.sma(short_prices, 10)
        assert len(sma) == 0
        
        rsi = TechnicalIndicators.rsi(short_prices, 14)
        assert len(rsi) == 0

class TestIndicatorAnalyzer:
    """Test suite for IndicatorAnalyzer"""
    
    def test_analyze_rsi_oversold(self):
        """Test RSI analysis for oversold condition"""
        rsi_values = [25, 20, 15]
        result = IndicatorAnalyzer.analyze_rsi(rsi_values, 100.0)
        
        assert result.signal == "BUY"
        assert result.strength > 0
        assert result.name == "RSI"
    
    def test_analyze_rsi_overbought(self):
        """Test RSI analysis for overbought condition"""
        rsi_values = [75, 80, 85]
        result = IndicatorAnalyzer.analyze_rsi(rsi_values, 100.0)
        
        assert result.signal == "SELL"
        assert result.strength > 0
    
    def test_analyze_rsi_neutral(self):
        """Test RSI analysis for neutral condition"""
        rsi_values = [45, 50, 55]
        result = IndicatorAnalyzer.analyze_rsi(rsi_values, 100.0)
        
        assert result.signal == "HOLD"
    
    def test_composite_signal_generation(self):
        """Test composite signal generation"""
        from indicators.technical_indicators import IndicatorResult
        
        indicators = [
            IndicatorResult("RSI", [75], "BUY", 80),
            IndicatorResult("MACD", [0.5], "BUY", 70),
            IndicatorResult("BB", [100, 105, 95], "HOLD", 50)
        ]
        
        composite = IndicatorAnalyzer.get_composite_signal(indicators)
        
        assert "signal" in composite
        assert "strength" in composite
        assert "confidence" in composite
        assert composite["signal"] in ["BUY", "SELL", "HOLD"]