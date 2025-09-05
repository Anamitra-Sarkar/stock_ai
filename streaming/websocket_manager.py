"""
Real-time WebSocket streaming for enterprise stock data
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import random
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Thread
import time

from cache.redis_cache import cache_manager
from agents.prediction_agent import PredictionAgent

class StreamingManager:
    """Enterprise-grade real-time data streaming manager"""
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.active_subscriptions = {}  # {room_id: {symbols: set(), clients: set()}}
        self.streaming_threads = {}
        self.prediction_agent = PredictionAgent()
        self.is_streaming = False
        
        # Market simulation parameters
        self.market_volatility = 0.02  # 2% base volatility
        self.price_cache = {}
        
        print("🚀 Real-time streaming manager initialized")
    
    def start_streaming(self):
        """Start background streaming threads"""
        if not self.is_streaming:
            self.is_streaming = True
            
            # Start market data simulation thread
            market_thread = Thread(target=self._simulate_market_data, daemon=True)
            market_thread.start()
            
            # Start prediction updates thread  
            prediction_thread = Thread(target=self._update_predictions, daemon=True)
            prediction_thread.start()
            
            print("✅ Real-time streaming started")
    
    def stop_streaming(self):
        """Stop all streaming"""
        self.is_streaming = False
        print("⏹️  Real-time streaming stopped")
    
    def subscribe_to_symbol(self, client_id: str, symbols: List[str], room_id: str = 'default'):
        """Subscribe client to real-time updates for symbols"""
        if room_id not in self.active_subscriptions:
            self.active_subscriptions[room_id] = {
                'symbols': set(),
                'clients': set()
            }
        
        self.active_subscriptions[room_id]['symbols'].update(symbols)
        self.active_subscriptions[room_id]['clients'].add(client_id)
        
        print(f"📡 Client {client_id} subscribed to {symbols} in room {room_id}")
        return True
    
    def unsubscribe_client(self, client_id: str, room_id: str = 'default'):
        """Unsubscribe client from updates"""
        if room_id in self.active_subscriptions:
            self.active_subscriptions[room_id]['clients'].discard(client_id)
            
            # Clean up empty rooms
            if len(self.active_subscriptions[room_id]['clients']) == 0:
                del self.active_subscriptions[room_id]
                
        print(f"📡 Client {client_id} unsubscribed from room {room_id}")
    
    def _simulate_market_data(self):
        """Simulate real-time market data updates"""
        while self.is_streaming:
            try:
                # Get all actively watched symbols
                all_symbols = set()
                for room_data in self.active_subscriptions.values():
                    all_symbols.update(room_data['symbols'])
                
                if not all_symbols:
                    time.sleep(1)
                    continue
                
                # Generate price updates
                updates = {}
                current_time = datetime.now()
                
                for symbol in all_symbols:
                    # Get or initialize price
                    if symbol not in self.price_cache:
                        self.price_cache[symbol] = self._get_base_price(symbol)
                    
                    # Simulate price movement
                    current_price = self.price_cache[symbol]
                    price_change = random.gauss(0, self.market_volatility)
                    new_price = current_price * (1 + price_change)
                    new_price = max(1.0, new_price)  # Ensure positive price
                    
                    # Calculate metrics
                    change_percent = (new_price - current_price) / current_price * 100
                    volume = random.randint(100000, 10000000)
                    
                    updates[symbol] = {
                        'symbol': symbol,
                        'price': round(new_price, 2),
                        'previous_price': round(current_price, 2),
                        'change': round(new_price - current_price, 2),
                        'change_percent': round(change_percent, 2),
                        'volume': volume,
                        'timestamp': current_time.isoformat(),
                        'trend': 'up' if change_percent > 0 else 'down' if change_percent < 0 else 'neutral'
                    }
                    
                    self.price_cache[symbol] = new_price
                
                # Broadcast updates to subscribed rooms
                for room_id, room_data in self.active_subscriptions.items():
                    room_updates = {symbol: updates[symbol] 
                                  for symbol in room_data['symbols'] 
                                  if symbol in updates}
                    
                    if room_updates:
                        self.socketio.emit('market_data', room_updates, room=room_id)
                
                # Update frequency: 2-5 seconds for realistic feel
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                print(f"❌ Error in market data simulation: {e}")
                time.sleep(5)
    
    def _update_predictions(self):
        """Update ML predictions periodically"""
        while self.is_streaming:
            try:
                # Get all actively watched symbols
                all_symbols = set()
                for room_data in self.active_subscriptions.values():
                    all_symbols.update(room_data['symbols'])
                
                if not all_symbols:
                    time.sleep(30)
                    continue
                
                # Update predictions (less frequent than price data)
                for symbol in all_symbols:
                    if symbol in self.price_cache:
                        current_price = self.price_cache[symbol]
                        
                        # Get updated prediction
                        prediction = asyncio.run(
                            self.prediction_agent.predict_trend(symbol, current_price)
                        )
                        
                        # Broadcast prediction update
                        prediction_data = {
                            'symbol': symbol,
                            'prediction': prediction,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Send to all rooms watching this symbol
                        for room_id, room_data in self.active_subscriptions.items():
                            if symbol in room_data['symbols']:
                                self.socketio.emit('prediction_update', prediction_data, room=room_id)
                
                # Update predictions every 30-60 seconds
                time.sleep(random.uniform(30, 60))
                
            except Exception as e:
                print(f"❌ Error updating predictions: {e}")
                time.sleep(30)
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol (mock data)"""
        # Mock base prices for different stocks
        base_prices = {
            'AAPL': 175.50,
            'GOOGL': 139.20,
            'MSFT': 378.85,
            'AMZN': 145.30,
            'TSLA': 248.50,
            'NVDA': 461.20,
            'META': 506.75,
            'JPM': 147.80,
            'BAC': 29.45,
            'WMT': 159.30,
            'JNJ': 156.78,
            'PG': 167.32
        }
        
        return base_prices.get(symbol, random.uniform(50, 300))
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get current market summary"""
        if not self.price_cache:
            return {'message': 'No active market data'}
        
        total_symbols = len(self.price_cache)
        positive_movers = sum(1 for price in self.price_cache.values() if price > 100)  # Simplified
        
        return {
            'active_symbols': total_symbols,
            'positive_movers': positive_movers,
            'negative_movers': total_symbols - positive_movers,
            'last_update': datetime.now().isoformat(),
            'market_status': 'STREAMING' if self.is_streaming else 'CLOSED'
        }
    
    def get_symbol_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a symbol"""
        if symbol not in self.price_cache:
            return None
        
        current_price = self.price_cache[symbol]
        
        # Generate additional metrics
        return {
            'symbol': symbol,
            'current_price': current_price,
            'day_high': current_price * random.uniform(1.01, 1.05),
            'day_low': current_price * random.uniform(0.95, 0.99),
            'opening_price': current_price * random.uniform(0.98, 1.02),
            'volume': random.randint(1000000, 50000000),
            'market_cap': current_price * random.randint(100000000, 3000000000),
            'pe_ratio': random.uniform(15, 35),
            'timestamp': datetime.now().isoformat()
        }
    
    def broadcast_alert(self, alert: Dict[str, Any], room_id: str = 'default'):
        """Broadcast trading alert to subscribers"""
        alert_data = {
            'type': 'alert',
            'alert': alert,
            'timestamp': datetime.now().isoformat()
        }
        
        self.socketio.emit('trading_alert', alert_data, room=room_id)
        print(f"🚨 Alert broadcasted to room {room_id}: {alert['message']}")
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        total_clients = sum(len(room['clients']) for room in self.active_subscriptions.values())
        total_symbols = len(set().union(*[room['symbols'] for room in self.active_subscriptions.values()]))
        
        return {
            'total_clients': total_clients,
            'active_rooms': len(self.active_subscriptions),
            'total_symbols': total_symbols,
            'streaming_active': self.is_streaming,
            'rooms': {room_id: {
                'clients': len(room['clients']),
                'symbols': len(room['symbols'])
            } for room_id, room in self.active_subscriptions.items()}
        }