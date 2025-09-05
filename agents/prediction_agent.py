import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class PredictionAgent:
    """
    Prediction Agent: Forecasts prices and growth potential using a machine learning model.
    This is a simplified example using Linear Regression. For a real-world scenario,
    you would use more sophisticated models like LSTMs (from TensorFlow/PyTorch)
    and train them on vast amounts of historical data.
    """
    def __init__(self):
        self.models = {}
        self._train_initial_models()

    def _generate_mock_data(self, ticker):
        """Generates mock historical data for training purposes."""
        np.random.seed(hash(ticker) % (2**32 - 1)) # Seed for consistent results
        dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=100))
        price = 100 + np.random.randn(100).cumsum()
        volume = np.random.randint(1_000_000, 10_000_000, size=100)
        # Create features: Moving Averages
        df = pd.DataFrame({'price': price, 'volume': volume}, index=dates)
        df['ma5'] = df['price'].rolling(window=5).mean()
        df['ma20'] = df['price'].rolling(window=20).mean()
        df['target'] = df['price'].shift(-1) # Predict next day's price
        return df.dropna()

    def _train_initial_models(self):
        """Trains a model for each stock on the mock historical data."""
        tickers = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT']
        for ticker in tickers:
            df = self._generate_mock_data(ticker)

            X = df[['price', 'volume', 'ma5', 'ma20']]
            y = df['target']

            # Note: In a real scenario, you wouldn't split randomly for time-series.
            # This is just for demonstration.
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models[ticker] = model

    def predict_trend(self, ticker, current_price):
        """
        Predicts the future trend and confidence for a stock.
        """
        if ticker not in self.models:
            return {"trend": "neutral", "confidence": 50, "predicted_price": current_price}

        # For prediction, we would need live feature data (volume, MAs).
        # We will simulate this for the prototype.
        mock_features = {
            'price': [current_price],
            'volume': [np.random.randint(1_000_000, 10_000_000)],
            'ma5': [current_price * 1.01],
            'ma20': [current_price * 0.98]
        }
        features_df = pd.DataFrame(mock_features)

        predicted_price = self.models[ticker].predict(features_df)[0]

        trend = "up" if predicted_price > current_price else "down"

        # Confidence is simulated here. A real model would output this, perhaps
        # based on the variance of predictions or model accuracy.
        confidence = random.randint(75, 95)

        return {
            "trend": trend,
            "confidence": confidence,
            "predicted_price": round(predicted_price, 2)
        }
