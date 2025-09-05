"""
Unit tests for the enterprise prediction agent
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from agents.prediction_agent import PredictionAgent

class TestPredictionAgent:
    """Test suite for PredictionAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create prediction agent instance"""
        return PredictionAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert hasattr(agent, 'models')
        assert hasattr(agent, 'lstm_predictors')
        assert len(agent.models) > 0 or len(agent.lstm_predictors) > 0
    
    def test_generate_mock_data(self, agent):
        """Test mock data generation"""
        ticker = "AAPL"
        data = agent._generate_mock_data(ticker)
        
        assert len(data) > 0
        assert all(key in data[0] for key in ['timestamp', 'symbol', 'close_price', 'volume'])
        assert data[0]['symbol'] == ticker
        assert all(d['close_price'] > 0 for d in data)
        assert all(d['volume'] > 0 for d in data)
    
    @pytest.mark.asyncio
    async def test_predict_trend_basic(self, agent):
        """Test basic trend prediction"""
        result = await agent.predict_trend('AAPL', 150.0)
        
        assert 'trend' in result
        assert 'confidence' in result
        assert 'predicted_price' in result
        assert result['trend'] in ['up', 'down', 'neutral']
        assert 0 <= result['confidence'] <= 100
        assert result['predicted_price'] > 0
    
    @pytest.mark.asyncio
    async def test_predict_trend_with_cache(self, agent):
        """Test prediction with caching"""
        # First call
        result1 = await agent.predict_trend('AAPL', 150.0, use_cache=True)
        
        # Second call should use cache
        result2 = await agent.predict_trend('AAPL', 150.0, use_cache=True)
        
        assert result1['ticker'] == result2['ticker']
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_analysis(self, agent):
        """Test multi-timeframe analysis"""
        result = await agent.get_multi_timeframe_analysis('AAPL', 150.0)
        
        assert isinstance(result, dict)
        assert '1d' in result
        assert 'timeframe' in result['1d']
    
    def test_model_status(self, agent):
        """Test model status reporting"""
        status = agent.get_model_status()
        
        assert 'lstm_models_loaded' in status
        assert 'legacy_models_loaded' in status
        assert 'available_symbols' in status
        assert isinstance(status['available_symbols'], list)
    
    def test_legacy_prediction_fallback(self, agent):
        """Test fallback to legacy prediction"""
        result = agent._legacy_prediction('UNKNOWN_TICKER', 100.0)
        
        assert 'trend' in result
        assert 'model_type' in result
        assert result['predicted_price'] > 0
    
    def test_calculate_composite_confidence(self, agent):
        """Test composite confidence calculation"""
        prediction = {
            'trend': 'up',
            'base_confidence': 80
        }
        technical = {
            'technical_signal': 'BUY',
            'technical_confidence': 75
        }
        
        confidence = agent._calculate_composite_confidence(prediction, technical)
        
        assert isinstance(confidence, float)
        assert 25 <= confidence <= 95