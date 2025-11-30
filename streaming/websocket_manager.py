"""
Real-time WebSocket streaming for enterprise stock data
"""
import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Thread

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
    
    def _deterministic_hash(self, symbol: str, seed: int = 0) -> float:
        """Generate deterministic pseudo-random value using hash"""
        combined = f"{symbol}_{seed}_{int(time.time() // 60)}"  # Change every minute
        hash_val = int(hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        return hash_val / (2**32 - 1)
    
    def _deterministic_gauss(self, symbol: str, mean: float = 0, sigma: float = 1, seed: int = 0) -> float:
        """Generate deterministic gaussian-like value using hash"""
        # Use Box-Muller transform approximation with hash
        u1 = self._deterministic_hash(symbol, seed)
        u2 = self._deterministic_hash(symbol, seed + 1)
        if u1 < 1e-10:
            u1 = 1e-10
        z0 = sigma * (-2 * (u1**0.5)) + mean
        return max(-3*sigma, min(3*sigma, z0))  # Clamp to reasonable range
    
    def _deterministic_randint(self, symbol: str, min_val: int, max_val: int, seed: int = 0) -> int:
        """Generate deterministic integer in range using hash"""
        hash_val = self._deterministic_hash(symbol, seed)
        return int(min_val + hash_val * (max_val - min_val))
    
    def _deterministic_uniform(self, symbol: str, min_val: float, max_val: float, seed: int = 0) -> float:
        """Generate deterministic uniform value using hash"""
        hash_val = self._deterministic_hash(symbol, seed)
        return min_val + hash_val * (max_val - min_val)
    
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
                    price_change = self._deterministic_gauss(symbol, 0, self.market_volatility)
                    new_price = current_price * (1 + price_change)
                    new_price = max(1.0, new_price)  # Ensure positive price
                    
                    # Calculate metrics
                    change_percent = (new_price - current_price) / current_price * 100
                    volume = self._deterministic_randint(symbol, 100000, 10000000, seed=1)
                    
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
                sleep_time = 2 + (int(time.time()) % 100) / 100.0 * 3  # Deterministic 2-5 second range
                time.sleep(sleep_time)
                
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
                sleep_time = 30 + (int(time.time()) % 100) / 100.0 * 30  # Deterministic 30-60 second range
                time.sleep(sleep_time)
                
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
        
        return base_prices.get(symbol, self._deterministic_uniform(symbol, 50, 300, seed=999))
    
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
            'day_high': current_price * self._deterministic_uniform(symbol, 1.01, 1.05, seed=2),
            'day_low': current_price * self._deterministic_uniform(symbol, 0.95, 0.99, seed=3),
            'opening_price': current_price * self._deterministic_uniform(symbol, 0.98, 1.02, seed=4),
            'volume': self._deterministic_randint(symbol, 1000000, 50000000, seed=5),
            'market_cap': current_price * self._deterministic_randint(symbol, 100000000, 3000000000, seed=6),
            'pe_ratio': self._deterministic_uniform(symbol, 15, 35, seed=7),
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
        
        # Efficient set union using itertools.chain
        all_symbols = set()
        for room in self.active_subscriptions.values():
            all_symbols.update(room['symbols'])
        total_symbols = len(all_symbols)
        
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