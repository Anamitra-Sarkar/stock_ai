import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import enterprise modules
from ml_models.lstm_predictor import LSTMPredictor
from cache.redis_cache import cache_manager
from indicators.technical_indicators import TechnicalIndicators, IndicatorAnalyzer

class PredictionAgent:
    """
    Enterprise Prediction Agent: Advanced price forecasting using LSTM neural networks
    with technical indicators, caching, and fallback mechanisms.
    """
    def __init__(self):
        self.models = {}  # Legacy models for backward compatibility
        self.lstm_predictors = {}  # LSTM models for each symbol
        self.technical_analyzer = TechnicalIndicators()
        self.indicator_analyzer = IndicatorAnalyzer()
        self._train_initial_models()
    
    def _generate_mock_data(self, ticker: str) -> pd.DataFrame:
        """Enhanced mock data generation with more realistic patterns"""
        np.random.seed(hash(ticker) % (2**32 - 1))
        
        # Generate more realistic price data
        n_points = 200  # Increased data points
        base_price = 100
        volatility = 0.02  # 2% daily volatility
        trend = np.random.uniform(-0.0005, 0.0005)  # Small trend
        
        prices = [base_price]
        for i in range(n_points - 1):
            # Mean-reverting random walk with trend
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(1.0, new_price))  # Ensure positive prices
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=n_points)
        dates = [start_date + timedelta(days=i) for i in range(n_points)]
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005)) if i > 0 else close
            volume = int(np.random.lognormal(15, 1))  # Log-normal volume distribution
            
            data.append({
                'timestamp': date,
                'symbol': ticker,
                'open_price': open_price,
                'high_price': high,
                'low_price': low,
                'close_price': close,
                'volume': volume
            })
        
        return data
    
    def _train_initial_models(self):
        """Initialize both legacy and LSTM models"""
        tickers = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'META', 'JPM']
        
        for ticker in tickers:
            try:
                # Generate training data
                training_data = self._generate_mock_data(ticker)
                
                # Create and train LSTM predictor
                lstm_predictor = LSTMPredictor(ticker)
                
                # Train the model (this will fallback to LinearRegression if TensorFlow unavailable)
                metrics = lstm_predictor.train(training_data)
                
                if 'error' not in metrics:
                    self.lstm_predictors[ticker] = lstm_predictor
                    print(f"✅ {ticker} model trained: {metrics['model_type']} (Validation MAE: {metrics.get('validation_mae', 'N/A'):.4f})")
                else:
                    print(f"❌ Failed to train {ticker}: {metrics['error']}")
                    
                # Legacy model for backward compatibility
                self._train_legacy_model(ticker, training_data)
                
            except Exception as e:
                print(f"❌ Error training models for {ticker}: {e}")
    
    def _train_legacy_model(self, ticker: str, training_data: List[Dict[str, Any]]):
        """Train legacy linear regression model"""
        try:
            df = pd.DataFrame(training_data)
            df['ma5'] = df['close_price'].rolling(window=5).mean()
            df['ma20'] = df['close_price'].rolling(window=20).mean()
            df['target'] = df['close_price'].shift(-1)
            df = df.dropna()
            
            if len(df) > 20:
                X = df[['close_price', 'volume', 'ma5', 'ma20']]
                y = df['target']
                
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                self.models[ticker] = model
                
        except Exception as e:
            print(f"⚠️  Legacy model training failed for {ticker}: {e}")
    
    async def predict_trend(self, ticker: str, current_price: float, 
                          use_cache: bool = True) -> Dict[str, Any]:
        """
        Advanced trend prediction using LSTM, technical indicators, and caching
        """
        # Check cache first
        if use_cache:
            cached_prediction = await cache_manager.get_prediction(ticker, 'LSTM')
            if cached_prediction:
                return cached_prediction
        
        try:
            # Use LSTM predictor if available
            if ticker in self.lstm_predictors:
                prediction_result = await self._lstm_prediction(ticker, current_price)
            else:
                # Fallback to legacy prediction
                prediction_result = self._legacy_prediction(ticker, current_price)
            
            # Add technical analysis
            technical_signals = await self._get_technical_signals(ticker, current_price)
            prediction_result.update(technical_signals)
            
            # Calculate composite confidence
            prediction_result['confidence'] = self._calculate_composite_confidence(
                prediction_result, technical_signals
            )
            
            # Cache the result
            if use_cache:
                await cache_manager.cache_prediction(ticker, 'LSTM', prediction_result, 15)
            
            return prediction_result
            
        except Exception as e:
            print(f"❌ Prediction failed for {ticker}: {e}")
            return self._fallback_prediction(ticker, current_price)
    
    async def _lstm_prediction(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Generate LSTM-based prediction"""
        predictor = self.lstm_predictors[ticker]
        
        # Generate recent mock data for prediction (in production, use real data)
        recent_data = self._generate_mock_data(ticker)[-60:]  # Last 60 days
        recent_data[-1]['close_price'] = current_price  # Update with current price
        
        # Get LSTM prediction
        result = predictor.predict(recent_data, prediction_days=1)
        
        # Determine trend
        predicted_price = result['predicted_price']
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > 0.02:
            trend = "up"
        elif price_change < -0.02:
            trend = "down"
        else:
            trend = "neutral"
        
        return {
            "ticker": ticker,
            "trend": trend,
            "predicted_price": round(predicted_price, 2),
            "current_price": current_price,
            "price_change_percent": round(price_change * 100, 2),
            "model_type": result['model_type'],
            "base_confidence": result['confidence'],
            "volatility": result.get('volatility', 0.02),
            "prediction_horizon": "1_day"
        }
    
    def _legacy_prediction(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Legacy linear regression prediction"""
        if ticker not in self.models:
            return self._fallback_prediction(ticker, current_price)
        
        # Mock features (in production, calculate from real data)
        mock_features = {
            'close_price': [current_price],
            'volume': [np.random.randint(1_000_000, 10_000_000)],
            'ma5': [current_price * 1.01],
            'ma20': [current_price * 0.98]
        }
        features_df = pd.DataFrame(mock_features)
        
        predicted_price = self.models[ticker].predict(features_df)[0]
        trend = "up" if predicted_price > current_price else "down"
        price_change = (predicted_price - current_price) / current_price
        
        return {
            "ticker": ticker,
            "trend": trend,
            "predicted_price": round(predicted_price, 2),
            "current_price": current_price,
            "price_change_percent": round(price_change * 100, 2),
            "model_type": "LinearRegression",
            "base_confidence": random.randint(65, 85),
            "volatility": 0.02,
            "prediction_horizon": "1_day"
        }
    
    async def _get_technical_signals(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Get technical analysis signals"""
        try:
            # Check cache for technical indicators
            cached_indicators = await cache_manager.get_technical_indicators(ticker, '1d')
            if cached_indicators:
                return cached_indicators
            
            # Generate mock price history for technical analysis
            price_history = self._generate_mock_data(ticker)
            prices = [d['close_price'] for d in price_history]
            high_prices = [d['high_price'] for d in price_history]
            low_prices = [d['low_price'] for d in price_history]
            volumes = [d['volume'] for d in price_history]
            
            # Calculate technical indicators
            rsi = self.technical_analyzer.rsi(prices, 14)
            macd_line, macd_signal, macd_hist = self.technical_analyzer.macd(prices)
            bb_upper, bb_middle, bb_lower = self.technical_analyzer.bollinger_bands(prices)
            
            # Analyze signals
            signals = []
            if rsi:
                rsi_signal = self.indicator_analyzer.analyze_rsi(rsi, current_price)
                signals.append(rsi_signal)
            
            if macd_hist:
                macd_signal_analysis = self.indicator_analyzer.analyze_macd(
                    macd_line, macd_signal, macd_hist
                )
                signals.append(macd_signal_analysis)
            
            if bb_upper and bb_lower:
                bb_signal = self.indicator_analyzer.analyze_bollinger_bands(
                    prices, bb_upper, bb_middle, bb_lower
                )
                signals.append(bb_signal)
            
            # Get composite signal
            composite = self.indicator_analyzer.get_composite_signal(signals)
            
            technical_result = {
                "technical_signal": composite["signal"],
                "technical_strength": composite["strength"],
                "technical_confidence": composite["confidence"],
                "indicators_analyzed": composite["indicators_count"],
                "rsi_current": rsi[-1] if rsi else None,
                "macd_histogram": macd_hist[-1] if macd_hist else None,
                "bollinger_position": "middle" if bb_middle else None
            }
            
            # Cache technical indicators
            await cache_manager.cache_technical_indicators(ticker, '1d', technical_result, 15)
            
            return technical_result
            
        except Exception as e:
            print(f"⚠️  Technical analysis failed for {ticker}: {e}")
            return {
                "technical_signal": "HOLD",
                "technical_strength": 30.0,
                "technical_confidence": 50.0,
                "indicators_analyzed": 0
            }
    
    def _calculate_composite_confidence(self, prediction: Dict[str, Any], 
                                      technical: Dict[str, Any]) -> float:
        """Calculate composite confidence from multiple sources"""
        base_confidence = prediction.get('base_confidence', 50)
        technical_confidence = technical.get('technical_confidence', 50)
        
        # Weight the confidences
        model_weight = 0.6
        technical_weight = 0.4
        
        # Boost confidence if signals agree
        model_signal = prediction.get('trend', 'neutral')
        technical_signal = technical.get('technical_signal', 'HOLD')
        
        agreement_boost = 0
        if (model_signal == 'up' and technical_signal == 'BUY') or \
           (model_signal == 'down' and technical_signal == 'SELL'):
            agreement_boost = 10
        
        composite_confidence = (
            base_confidence * model_weight + 
            technical_confidence * technical_weight + 
            agreement_boost
        )
        
        return min(95, max(25, round(composite_confidence, 1)))
    
    def _fallback_prediction(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Fallback prediction when all else fails"""
        return {
            "ticker": ticker,
            "trend": "neutral",
            "confidence": 50,
            "predicted_price": current_price,
            "current_price": current_price,
            "price_change_percent": 0.0,
            "model_type": "FALLBACK",
            "technical_signal": "HOLD",
            "technical_strength": 30.0,
            "error": "Model not available"
        }
    
    async def get_multi_timeframe_analysis(self, ticker: str, 
                                         current_price: float) -> Dict[str, Any]:
        """Get predictions across multiple timeframes"""
        timeframes = ['1d', '1w', '1m']
        results = {}
        
        for tf in timeframes:
            try:
                # In production, this would use different models trained on different timeframes
                prediction = await self.predict_trend(ticker, current_price, use_cache=True)
                
                # Adjust confidence based on timeframe
                if tf == '1w':
                    prediction['confidence'] *= 0.9  # Slightly less confident for weekly
                elif tf == '1m':
                    prediction['confidence'] *= 0.8  # Less confident for monthly
                
                prediction['timeframe'] = tf
                results[tf] = prediction
                
            except Exception as e:
                results[tf] = {'error': str(e), 'timeframe': tf}
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all prediction models"""
        return {
            'lstm_models_loaded': len(self.lstm_predictors),
            'legacy_models_loaded': len(self.models),
            'available_symbols': list(set(list(self.lstm_predictors.keys()) + list(self.models.keys()))),
            'cache_available': cache_manager is not None,
            'total_models': len(self.lstm_predictors) + len(self.models)
        }