"""
Enterprise-grade technical indicators for advanced market analysis
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class IndicatorResult:
    """Container for technical indicator results"""
    name: str
    values: List[float]
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-100 signal strength

class TechnicalIndicators:
    """Comprehensive technical analysis indicators suite"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average using efficient rolling window"""
        if len(prices) < period:
            return []
        
        # Use rolling sum for O(n) complexity instead of O(n*period)
        window_sum = sum(prices[:period])
        sma_values = [window_sum / period]
        
        for i in range(period, len(prices)):
            window_sum += prices[i] - prices[i - period]
            sma_values.append(window_sum / period)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = []
        
        # Start with SMA for the first value
        ema = sum(prices[:period]) / period
        ema_values.append(ema)
        
        # Calculate EMA for remaining values
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        gains = [max(delta, 0) for delta in deltas]
        losses = [-min(delta, 0) for delta in deltas]
        
        # Calculate initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(deltas)):
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
            
            # Update averages (Wilder's smoothing)
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow_period:
            return [], [], []
        
        ema_fast = TechnicalIndicators.ema(prices, fast_period)
        ema_slow = TechnicalIndicators.ema(prices, slow_period)
        
        # Align the EMAs (slow EMA starts later)
        start_idx = slow_period - fast_period
        ema_fast = ema_fast[start_idx:]
        
        # Calculate MACD line
        macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
        
        # Calculate signal line (EMA of MACD line)
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Calculate histogram (align signal line with MACD line)
        macd_aligned = macd_line[signal_period - 1:]
        histogram = [macd - signal for macd, signal in zip(macd_aligned, signal_line)]
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, 
                       std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands using efficient rolling calculations"""
        if len(prices) < period:
            return [], [], []
        
        # Convert to numpy array for efficient calculations
        prices_arr = np.array(prices)
        n = len(prices_arr)
        
        # Pre-allocate result arrays
        upper_band = []
        middle_band = []
        lower_band = []
        
        # Calculate initial window stats
        window = prices_arr[:period]
        window_sum = window.sum()
        window_sq_sum = (window ** 2).sum()
        
        for i in range(period - 1, n):
            if i > period - 1:
                # Update rolling sums efficiently
                old_val = prices_arr[i - period]
                new_val = prices_arr[i]
                window_sum += new_val - old_val
                window_sq_sum += new_val ** 2 - old_val ** 2
            
            # Calculate mean and std from rolling sums
            mean = window_sum / period
            variance = (window_sq_sum / period) - (mean ** 2)
            std = np.sqrt(max(0, variance))  # Ensure non-negative due to floating point
            
            middle_band.append(mean)
            upper_band.append(mean + (std * std_dev))
            lower_band.append(mean - (std * std_dev))
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def stochastic_oscillator(high: List[float], low: List[float], close: List[float],
                            k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator (%K and %D)"""
        if len(close) < k_period:
            return [], []
        
        k_values = []
        
        for i in range(k_period - 1, len(close)):
            window_high = max(high[i - k_period + 1:i + 1])
            window_low = min(low[i - k_period + 1:i + 1])
            
            if window_high == window_low:
                k_values.append(50)  # Avoid division by zero
            else:
                k = ((close[i] - window_low) / (window_high - window_low)) * 100
                k_values.append(k)
        
        # %D is SMA of %K
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def atr(high: List[float], low: List[float], close: List[float], 
            period: int = 14) -> List[float]:
        """Average True Range"""
        if len(close) < 2:
            return []
        
        true_ranges = []
        
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return []
        
        # First ATR is simple average
        atr_values = [sum(true_ranges[:period]) / period]
        
        # Subsequent ATRs use Wilder's smoothing
        for i in range(period, len(true_ranges)):
            atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
            atr_values.append(atr)
        
        return atr_values
    
    @staticmethod
    def williams_r(high: List[float], low: List[float], close: List[float],
                  period: int = 14) -> List[float]:
        """Williams %R"""
        if len(close) < period:
            return []
        
        williams_values = []
        
        for i in range(period - 1, len(close)):
            window_high = max(high[i - period + 1:i + 1])
            window_low = min(low[i - period + 1:i + 1])
            
            if window_high == window_low:
                williams_values.append(-50)  # Avoid division by zero
            else:
                williams = ((window_high - close[i]) / (window_high - window_low)) * -100
                williams_values.append(williams)
        
        return williams_values
    
    @staticmethod
    def obv(close: List[float], volume: List[int]) -> List[float]:
        """On-Balance Volume"""
        if len(close) != len(volume) or len(close) < 2:
            return []
        
        obv_values = [0]  # Start with 0
        
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv_values.append(obv_values[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv_values.append(obv_values[-1] - volume[i])
            else:
                obv_values.append(obv_values[-1])
        
        return obv_values
    
    @staticmethod
    def cci(high: List[float], low: List[float], close: List[float],
           period: int = 20) -> List[float]:
        """Commodity Channel Index using efficient rolling calculations"""
        if len(close) < period:
            return []
        
        # Calculate Typical Price using numpy for efficiency
        typical_prices = np.array([(h + l + c) / 3 for h, l, c in zip(high, low, close)])
        
        n = len(typical_prices)
        cci_values = []
        
        # Use rolling sum for efficient SMA calculation
        window_sum = typical_prices[:period].sum()
        
        for i in range(period - 1, n):
            if i > period - 1:
                # Update rolling sum efficiently
                window_sum += typical_prices[i] - typical_prices[i - period]
            
            sma_tp = window_sum / period
            
            # Calculate Mean Absolute Deviation
            # Need full window access for MAD, but this is unavoidable
            window = typical_prices[i - period + 1:i + 1]
            mad = np.abs(window - sma_tp).sum() / period
            
            if mad == 0:
                cci_values.append(0)
            else:
                cci = (typical_prices[i] - sma_tp) / (0.015 * mad)
                cci_values.append(cci)
        
        return cci_values

class IndicatorAnalyzer:
    """Analyzer to generate trading signals from technical indicators"""
    
    @staticmethod
    def analyze_rsi(rsi_values: List[float], current_price: float) -> IndicatorResult:
        """Analyze RSI for trading signals"""
        if not rsi_values:
            return IndicatorResult("RSI", [], "HOLD", 0)
        
        current_rsi = rsi_values[-1]
        
        if current_rsi > 70:
            signal = "SELL"
            strength = min((current_rsi - 70) * 3.33, 100)  # Scale to 100
        elif current_rsi < 30:
            signal = "BUY"
            strength = min((30 - current_rsi) * 3.33, 100)  # Scale to 100
        else:
            signal = "HOLD"
            strength = abs(current_rsi - 50) * 2  # Distance from neutral
        
        return IndicatorResult("RSI", rsi_values, signal, strength)
    
    @staticmethod
    def analyze_macd(macd_line: List[float], signal_line: List[float], 
                    histogram: List[float]) -> IndicatorResult:
        """Analyze MACD for trading signals"""
        if len(histogram) < 2:
            return IndicatorResult("MACD", [], "HOLD", 0)
        
        current_hist = histogram[-1]
        prev_hist = histogram[-2]
        
        # MACD line above signal line and increasing
        if current_hist > 0 and current_hist > prev_hist:
            signal = "BUY"
            strength = min(abs(current_hist) * 50, 100)
        # MACD line below signal line and decreasing  
        elif current_hist < 0 and current_hist < prev_hist:
            signal = "SELL"
            strength = min(abs(current_hist) * 50, 100)
        else:
            signal = "HOLD"
            strength = abs(current_hist) * 25
        
        return IndicatorResult("MACD", histogram, signal, strength)
    
    @staticmethod
    def analyze_bollinger_bands(prices: List[float], upper: List[float], 
                               middle: List[float], lower: List[float]) -> IndicatorResult:
        """Analyze Bollinger Bands for trading signals"""
        if not prices or not upper or not lower:
            return IndicatorResult("Bollinger", [], "HOLD", 0)
        
        current_price = prices[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]
        
        band_width = current_upper - current_lower
        
        # Price near upper band - potential sell
        if current_price >= current_upper * 0.98:
            signal = "SELL"
            strength = ((current_price - current_middle) / (current_upper - current_middle)) * 100
        # Price near lower band - potential buy
        elif current_price <= current_lower * 1.02:
            signal = "BUY" 
            strength = ((current_middle - current_price) / (current_middle - current_lower)) * 100
        else:
            signal = "HOLD"
            strength = abs(current_price - current_middle) / band_width * 50
        
        return IndicatorResult("Bollinger", [current_upper, current_middle, current_lower], 
                             signal, min(strength, 100))
    
    @staticmethod
    def get_composite_signal(indicators: List[IndicatorResult]) -> Dict[str, any]:
        """Generate composite trading signal from multiple indicators"""
        if not indicators:
            return {"signal": "HOLD", "strength": 0, "confidence": 0}
        
        buy_signals = [ind for ind in indicators if ind.signal == "BUY"]
        sell_signals = [ind for ind in indicators if ind.signal == "SELL"]
        
        buy_strength = sum(ind.strength for ind in buy_signals) / len(indicators)
        sell_strength = sum(ind.strength for ind in sell_signals) / len(indicators)
        
        # Determine overall signal
        if buy_strength > sell_strength and buy_strength > 30:
            overall_signal = "BUY"
            overall_strength = buy_strength
        elif sell_strength > buy_strength and sell_strength > 30:
            overall_signal = "SELL" 
            overall_strength = sell_strength
        else:
            overall_signal = "HOLD"
            overall_strength = max(buy_strength, sell_strength)
        
        # Calculate confidence based on agreement
        agreement_count = len(buy_signals) if overall_signal == "BUY" else len(sell_signals)
        confidence = (agreement_count / len(indicators)) * 100
        
        return {
            "signal": overall_signal,
            "strength": round(overall_strength, 2),
            "confidence": round(confidence, 2),
            "indicators_count": len(indicators),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals)
        }