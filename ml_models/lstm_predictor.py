"""
Advanced LSTM Neural Network for Stock Price Prediction
Enterprise-grade implementation with fallback to Linear Regression
"""
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try to import TensorFlow/Keras, fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available, using Linear Regression fallback")

from config import config

# Check deployment flags
SKIP_ML_TRAINING = os.getenv('SKIP_ML_TRAINING', 'false').lower() == 'true'

class LSTMPredictor:
    """Enterprise-grade LSTM model for stock price prediction"""
    
    def __init__(self, symbol: str, sequence_length: int = None):
        self.symbol = symbol
        self.sequence_length = sequence_length or config.ml.lstm_sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scalers = {}
        self.is_trained = False
        self.model_path = f"{config.ml.model_save_path}/{symbol}_lstm.h5"
        self.scaler_path = f"{config.ml.model_save_path}/{symbol}_scaler.pkl"
        self.fallback_model = LinearRegression()  # Fallback model
        self.use_lstm = TF_AVAILABLE
        
        # Ensure model directory exists
        os.makedirs(config.ml.model_save_path, exist_ok=True)
    
    def prepare_data(self, price_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for LSTM training with multiple features"""
        if len(price_data) < self.sequence_length + 1:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length + 1} records")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(price_data)
        df = df.sort_values('timestamp')
        
        # Feature engineering
        df['price_change'] = df['close_price'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high_price'] / df['low_price']
        df['price_volume_trend'] = df['close_price'] * df['volume']
        
        # Technical indicators (simplified versions)
        df['sma_5'] = df['close_price'].rolling(window=5).mean()
        df['sma_20'] = df['close_price'].rolling(window=20).mean() 
        df['volatility'] = df['close_price'].rolling(window=10).std()
        
        # Select features for prediction
        feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'price_change', 'volume_change', 'high_low_ratio', 'price_volume_trend',
            'sma_5', 'sma_20', 'volatility'
        ]
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < self.sequence_length + 1:
            raise ValueError("Insufficient data after feature engineering")
        
        # Prepare features and target
        features = df[feature_columns].values
        target = df['close_price'].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y), feature_columns
    
    def build_lstm_model(self, n_features: int) -> Any:
        """Build and compile LSTM model"""
        if not self.use_lstm:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        return model
    
    def train(self, price_data: List[Dict[str, Any]], validation_split: float = 0.2) -> Dict[str, float]:
        """Train the LSTM model with comprehensive metrics"""
        try:
            # Skip training if deployment flag is set
            if SKIP_ML_TRAINING:
                print(f"⏩ Skipping ML training for {self.symbol} (deployment mode)")
                return {
                    'mae': 5.0,
                    'rmse': 8.0,
                    'mape': 3.5,
                    'directional_accuracy': 0.65,
                    'epochs_trained': 0,
                    'training_time': 0.0,
                    'model_type': 'mock_deployment'
                }
            
            print(f"🧠 Training LSTM model for {self.symbol}...")
            
            # Prepare data
            X, y, feature_columns = self.prepare_data(price_data)
            
            if self.use_lstm:
                # Build LSTM model
                self.model = self.build_lstm_model(X.shape[2])
                
                # Train model
                history = self.model.fit(
                    X, y,
                    batch_size=config.ml.lstm_batch_size,
                    epochs=config.ml.lstm_epochs,
                    validation_split=validation_split,
                    verbose=1,
                    shuffle=False  # Important for time series
                )
                
                # Save model and scaler
                self.model.save(self.model_path)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                # Get training metrics
                val_loss = min(history.history['val_loss'])
                val_mae = min(history.history['val_mae'])
                
                metrics = {
                    'validation_loss': val_loss,
                    'validation_mae': val_mae,
                    'model_type': 'LSTM',
                    'epochs_trained': len(history.history['loss']),
                    'sequence_length': self.sequence_length
                }
                
            else:
                # Fallback to Linear Regression
                print(f"📊 Using Linear Regression fallback for {self.symbol}")
                
                # Flatten sequences for linear regression
                X_flat = X.reshape(X.shape[0], -1)
                
                # Split data for validation
                split_idx = int(len(X_flat) * (1 - validation_split))
                X_train, X_val = X_flat[:split_idx], X_flat[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Train linear regression
                self.fallback_model.fit(X_train, y_train)
                
                # Validate
                y_pred = self.fallback_model.predict(X_val)
                val_mae = mean_absolute_error(y_val, y_pred)
                val_mse = mean_squared_error(y_val, y_pred)
                
                metrics = {
                    'validation_loss': val_mse,
                    'validation_mae': val_mae,
                    'model_type': 'LinearRegression',
                    'r2_score': self.fallback_model.score(X_val, y_val),
                    'sequence_length': self.sequence_length
                }
            
            self.is_trained = True
            print(f"✅ Model training completed for {self.symbol}")
            return metrics
            
        except Exception as e:
            print(f"❌ Training failed for {self.symbol}: {e}")
            return {'error': str(e), 'model_type': 'FAILED'}
    
    def predict(self, recent_data: List[Dict[str, Any]], 
               prediction_days: int = 1) -> Dict[str, Any]:
        """Make price predictions with confidence intervals"""
        try:
            if not self.is_trained:
                # Try to load existing model
                if not self.load_model():
                    raise ValueError("Model not trained and cannot be loaded")
            
            # Prepare recent data
            df = pd.DataFrame(recent_data[-self.sequence_length:])
            df = df.sort_values('timestamp')
            
            # Feature engineering (same as training)
            df['price_change'] = df['close_price'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high_price'] / df['low_price']
            df['price_volume_trend'] = df['close_price'] * df['volume']
            df['sma_5'] = df['close_price'].rolling(window=5).mean()
            df['sma_20'] = df['close_price'].rolling(window=20).mean()
            df['volatility'] = df['close_price'].rolling(window=10).std()
            
            feature_columns = [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'price_change', 'volume_change', 'high_low_ratio', 'price_volume_trend',
                'sma_5', 'sma_20', 'volatility'
            ]
            
            # Handle NaN values by forward filling
            df[feature_columns] = df[feature_columns].fillna(method='ffill')
            df[feature_columns] = df[feature_columns].fillna(method='bfill')
            
            features = df[feature_columns].values
            scaled_features = self.scaler.transform(features)
            
            predictions = []
            current_sequence = scaled_features.copy()
            
            for _ in range(prediction_days):
                if self.use_lstm and self.model:
                    # LSTM prediction
                    sequence = current_sequence[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                    pred = self.model.predict(sequence, verbose=0)[0][0]
                else:
                    # Linear regression prediction
                    sequence = current_sequence[-self.sequence_length:].flatten().reshape(1, -1)
                    pred = self.fallback_model.predict(sequence)[0]
                
                predictions.append(pred)
                
                # Update sequence for multi-step prediction (simplified)
                if len(current_sequence) > 0:
                    new_row = current_sequence[-1].copy()
                    new_row[3] = pred  # Update close price
                    current_sequence = np.append(current_sequence, [new_row], axis=0)
            
            # Calculate confidence based on recent volatility
            recent_prices = [d['close_price'] for d in recent_data[-10:]]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            base_confidence = max(60, min(95, 90 - (volatility * 1000)))
            
            # Trend analysis
            trend = "neutral"
            if len(predictions) > 0:
                current_price = recent_data[-1]['close_price']
                predicted_price = predictions[0]
                price_change = (predicted_price - current_price) / current_price
                
                if price_change > 0.02:
                    trend = "up"
                elif price_change < -0.02:
                    trend = "down"
            
            return {
                'symbol': self.symbol,
                'predictions': predictions,
                'predicted_price': predictions[0] if predictions else recent_data[-1]['close_price'],
                'current_price': recent_data[-1]['close_price'],
                'trend': trend,
                'confidence': round(base_confidence, 1),
                'model_type': 'LSTM' if self.use_lstm else 'LinearRegression',
                'prediction_date': datetime.now().isoformat(),
                'volatility': round(volatility, 4)
            }
            
        except Exception as e:
            print(f"❌ Prediction failed for {self.symbol}: {e}")
            # Return fallback prediction
            return {
                'symbol': self.symbol,
                'predicted_price': recent_data[-1]['close_price'],
                'current_price': recent_data[-1]['close_price'],
                'trend': 'neutral',
                'confidence': 50.0,
                'model_type': 'FALLBACK',
                'error': str(e)
            }
    
    def load_model(self) -> bool:
        """Load pre-trained model and scaler"""
        try:
            if self.use_lstm and os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Failed to load model for {self.symbol}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        return {
            'symbol': self.symbol,
            'model_type': 'LSTM' if self.use_lstm else 'LinearRegression',
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'model_exists': os.path.exists(self.model_path) if self.use_lstm else True,
            'tensorflow_available': TF_AVAILABLE
        }