# advanced_order_execution.py
"""
Sistema avanzado de ejecución de órdenes con algoritmos inteligentes
Incluye TWAP, VWAP, Iceberg orders, y smart routing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time
import random  # Añadido
from collections import defaultdict

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc, fok
    iceberg_quantity: Optional[float] = None
    twap_duration: Optional[int] = None  # minutes
    vwap_participation: Optional[float] = None  # % of volume
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    average_price: float = 0
    fees: float = 0
    slippage: float = 0
    created_at: datetime = None
    updated_at: datetime = None

class SmartOrderRouter:
    """Router inteligente de órdenes para mejor ejecución"""
    
    def __init__(self, broker_connections: Dict):
        self.brokers = broker_connections
        self.order_book_cache = {}
        self.liquidity_map = {}
        self.execution_costs = {}
        
    async def get_best_execution_venue(self, order: Order) -> Tuple[str, Dict]:
        """Determina el mejor venue para ejecutar la orden"""
        venues_analysis = {}
        
        for venue_name, broker in self.brokers.items():
            try:
                # Analizar liquidez
                liquidity = await self._analyze_liquidity(venue_name, order.symbol)
                
                # Estimar impacto en precio
                price_impact = self._estimate_price_impact(
                    order.quantity, 
                    liquidity
                )
                
                # Calcular costos
                fees = self._calculate_fees(venue_name, order)
                
                # Score total
                score = self._calculate_venue_score(
                    liquidity, 
                    price_impact, 
                    fees,
                    broker.get('latency', 1.0)  # Valor por defecto si no existe
                )
                
                venues_analysis[venue_name] = {
                    'score': score,
                    'liquidity': liquidity,
                    'price_impact': price_impact,
                    'fees': fees,
                    'estimated_price': liquidity.get('best_price', 0)
                }
                
            except Exception as e:
                print(f"Error analyzing venue {venue_name}: {e}")
        
        # Seleccionar mejor venue
        if not venues_analysis:
            # Fallback si no hay venues disponibles
            return list(self.brokers.keys())[0], {'score': 0, 'liquidity': {}, 'price_impact': 0, 'fees': 0}
            
        best_venue = max(venues_analysis.items(), key=lambda x: x[1]['score'])
        return best_venue[0], best_venue[1]
    
    async def _analyze_liquidity(self, venue: str, symbol: str) -> Dict:
        """Analiza liquidez en un venue específico"""
        # Obtener order book
        order_book = await self._fetch_order_book(venue, symbol)
        
        # Validar order book
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {
                'bid_depth': 0,
                'ask_depth': 0,
                'spread': 0,
                'spread_bps': 0,
                'best_bid': 0,
                'best_ask': 0,
                'best_price': 0
            }
        
        # Calcular métricas
        bid_depth = sum(level['size'] for level in order_book.get('bids', [])[:10])
        ask_depth = sum(level['size'] for level in order_book.get('asks', [])[:10])
        
        # Verificar que hay bids y asks
        if order_book['bids'] and order_book['asks']:
            spread = order_book['asks'][0]['price'] - order_book['bids'][0]['price']
            best_bid = order_book['bids'][0]['price']
            best_ask = order_book['asks'][0]['price']
            spread_bps = (spread / best_bid) * 10000 if best_bid > 0 else 0
        else:
            spread = 0
            spread_bps = 0
            best_bid = 0
            best_ask = 0
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'spread': spread,
            'spread_bps': spread_bps,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'best_price': best_ask if order_book.get('side') == 'buy' else best_bid
        }
    
    async def _fetch_order_book(self, venue: str, symbol: str) -> Dict:
        """Fetch order book simulado"""
        # En producción, esto haría una llamada real a la API del broker
        # Por ahora, simulamos un order book
        base_price = 100.0
        spread = 0.01
        
        return {
            'bids': [
                {'price': base_price - spread * (i + 1), 'size': random.randint(100, 1000)}
                for i in range(10)
            ],
            'asks': [
                {'price': base_price + spread * (i + 1), 'size': random.randint(100, 1000)}
                for i in range(10)
            ]
        }
    
    def _estimate_price_impact(self, quantity: float, liquidity: Dict) -> float:
        """Estima el impacto en precio de una orden"""
        # Modelo simple de impacto
        relevant_depth = liquidity['ask_depth'] if quantity > 0 else liquidity['bid_depth']
        
        if relevant_depth == 0:
            return float('inf')
        
        # Impacto no lineal
        size_ratio = quantity / relevant_depth
        base_impact = liquidity['spread_bps'] / 10000 if liquidity['spread_bps'] > 0 else 0.0001
        
        # Modelo: impact = base_impact * (1 + size_ratio^1.5)
        total_impact = base_impact * (1 + pow(size_ratio, 1.5))
        
        return total_impact
    
    def _calculate_fees(self, venue: str, order: Order) -> float:
        """Calcula fees para el venue"""
        # Fees típicos por venue (en producción vendría de configuración)
        base_fees = {
            'binance': 0.001,
            'coinbase': 0.0015,
            'kraken': 0.0016,
            'default': 0.002
        }
        
        fee_rate = base_fees.get(venue.lower(), base_fees['default'])
        
        # Ajustar por tipo de orden
        if order.order_type == OrderType.MARKET:
            fee_rate *= 1.5  # Market orders más caras
        
        return fee_rate
    
    def _calculate_venue_score(self, liquidity: Dict, price_impact: float, 
                              fees: float, latency: float) -> float:
        """Calcula score compuesto para un venue"""
        # Pesos para cada factor
        weights = {
            'liquidity': 0.3,
            'price_impact': 0.3,
            'fees': 0.25,
            'latency': 0.15
        }
        
        # Normalizar valores (0-1)
        liquidity_score = min(liquidity['bid_depth'] + liquidity['ask_depth'], 1000000) / 1000000
        impact_score = 1 / (1 + price_impact * 100)
        fee_score = 1 / (1 + fees * 1000)
        latency_score = 1 / (1 + latency / 10)
        
        # Score final
        total_score = (
            liquidity_score * weights['liquidity'] +
            impact_score * weights['price_impact'] +
            fee_score * weights['fees'] +
            latency_score * weights['latency']
        )
        
        return total_score

class AdvancedOrderExecutor:
    """Ejecutor avanzado con algoritmos de ejecución"""
    
    def __init__(self, router: SmartOrderRouter):
        self.router = router
        self.active_orders = {}
        self.execution_queue = queue.PriorityQueue()
        self.market_data_feed = None
        self.execution_thread = None
        self.running = False
        
    def start(self):
        """Inicia el ejecutor"""
        self.running = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.start()
    
    def stop(self):
        """Detiene el ejecutor"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
    
    def _execution_loop(self):
        """Loop principal de ejecución"""
        while self.running:
            try:
                # Procesar órdenes pendientes
                if not self.execution_queue.empty():
                    priority, order = self.execution_queue.get(timeout=1)
                    asyncio.run(self.execute_order(order))
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in execution loop: {e}")
    
    async def execute_order(self, order: Order) -> Dict:
        """Ejecuta una orden usando el algoritmo apropiado"""
        order.created_at = datetime.now()
        self.active_orders[order.id] = order
        
        # Seleccionar estrategia de ejecución
        if order.order_type == OrderType.TWAP:
            return await self._execute_twap(order)
        elif order.order_type == OrderType.VWAP:
            return await self._execute_vwap(order)
        elif order.order_type == OrderType.ICEBERG:
            return await self._execute_iceberg(order)
        else:
            return await self._execute_standard(order)
    
    async def _execute_twap(self, order: Order) -> Dict:
        """Ejecuta orden usando Time-Weighted Average Price"""
        duration_minutes = order.twap_duration or 30
        num_slices = min(duration_minutes, 20)  # Max 20 slices
        slice_size = order.quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        results = []
        total_filled = 0
        total_cost = 0
        
        for i in range(num_slices):
            if not self.running or order.status == OrderStatus.CANCELLED:
                break
            
            # Crear sub-orden
            sub_order = Order(
                id=f"{order.id}_twap_{i}",
                symbol=order.symbol,
                side=order.side,
                quantity=slice_size,
                order_type=OrderType.LIMIT,
                limit_price=await self._get_aggressive_price(order.symbol, order.side),
                time_in_force="ioc"
            )
            
            # Ejecutar slice
            result = await self._execute_single_order(sub_order)
            results.append(result)
            
            if result['filled_quantity'] > 0:
                total_filled += result['filled_quantity']
                total_cost += result['filled_quantity'] * result['average_price']
                
                # Actualizar orden principal
                order.filled_quantity = total_filled
                order.average_price = total_cost / total_filled if total_filled > 0 else 0
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Esperar hasta siguiente slice
            if i < num_slices - 1:
                await asyncio.sleep(interval_seconds)
        
        # Finalizar orden
        if total_filled >= order.quantity * 0.99:  # 99% filled
            order.status = OrderStatus.FILLED
        elif total_filled > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        return {
            'order_id': order.id,
            'status': order.status,
            'filled_quantity': total_filled,
            'average_price': order.average_price,
            'slices_executed': len(results)
        }
    
    async def _execute_vwap(self, order: Order) -> Dict:
        """Ejecuta orden usando Volume-Weighted Average Price"""
        participation_rate = order.vwap_participation or 0.1  # 10% del volumen
        
        # Obtener perfil de volumen histórico
        volume_profile = await self._get_volume_profile(order.symbol)
        
        results = []
        total_filled = 0
        total_cost = 0
        
        # Ejecutar siguiendo el perfil de volumen
        for period in volume_profile:
            if not self.running or order.status == OrderStatus.CANCELLED:
                break
            
            # Calcular tamaño basado en volumen esperado
            expected_volume = period['expected_volume']
            target_size = min(
                expected_volume * participation_rate,
                order.quantity - total_filled
            )
            
            if target_size <= 0:
                break
            
            # Ejecutar durante el período
            period_result = await self._execute_during_period(
                order, 
                target_size, 
                period['start_time'],
                period['end_time']
            )
            
            results.append(period_result)
            total_filled += period_result['filled_quantity']
            total_cost += period_result['filled_quantity'] * period_result['average_price']
            
            # Actualizar orden
            order.filled_quantity = total_filled
            order.average_price = total_cost / total_filled if total_filled > 0 else 0
        
        return {
            'order_id': order.id,
            'status': order.status,
            'filled_quantity': total_filled,
            'average_price': order.average_price,
            'periods_executed': len(results)
        }
    
    async def _execute_iceberg(self, order: Order) -> Dict:
        """Ejecuta orden iceberg (muestra solo parte del tamaño)"""
        visible_quantity = order.iceberg_quantity or order.quantity * 0.1
        
        results = []
        total_filled = 0
        total_cost = 0
        
        while total_filled < order.quantity and self.running:
            # Calcular siguiente chunk
            remaining = order.quantity - total_filled
            chunk_size = min(visible_quantity, remaining)
            
            # Crear orden visible
            visible_order = Order(
                id=f"{order.id}_iceberg_{len(results)}",
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_size,
                order_type=OrderType.LIMIT,
                limit_price=order.limit_price or await self._get_best_price(order.symbol, order.side),
                time_in_force="gtc"
            )
            
            # Ejecutar y monitorear
            result = await self._execute_and_monitor(visible_order)
            results.append(result)
            
            if result['filled_quantity'] > 0:
                total_filled += result['filled_quantity']
                total_cost += result['filled_quantity'] * result['average_price']
                
                # Actualizar orden principal
                order.filled_quantity = total_filled
                order.average_price = total_cost / total_filled
            
            # Pequeña pausa para no ser detectado
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return {
            'order_id': order.id,
            'status': order.status,
            'filled_quantity': total_filled,
            'average_price': order.average_price,
            'chunks_executed': len(results)
        }
    
    async def _execute_standard(self, order: Order) -> Dict:
        """Ejecuta orden estándar"""
        return await self._execute_single_order(order)
    
    async def _execute_single_order(self, order: Order) -> Dict:
        """Ejecuta una orden individual"""
        # Simulación de ejecución
        # En producción, esto enviaría la orden al broker
        
        # Simular fill parcial o completo
        fill_ratio = random.uniform(0.8, 1.0) if order.order_type == OrderType.LIMIT else 1.0
        filled_quantity = order.quantity * fill_ratio
        
        # Simular precio de ejecución
        base_price = 100.0  # En producción vendría del mercado
        slippage = random.uniform(-0.001, 0.001)
        execution_price = base_price * (1 + slippage)
        
        return {
            'order_id': order.id,
            'filled_quantity': filled_quantity,
            'average_price': execution_price,
            'fees': filled_quantity * execution_price * 0.001,
            'timestamp': datetime.now()
        }
    
    async def _execute_and_monitor(self, order: Order) -> Dict:
        """Ejecuta y monitorea una orden hasta completarse"""
        # Por ahora, simplemente ejecutamos
        return await self._execute_single_order(order)
    
    async def _execute_during_period(self, order: Order, target_size: float,
                                   start_time: datetime, end_time: datetime) -> Dict:
        """Ejecuta orden durante un período específico"""
        # Simplificado - ejecutar el target size
        sub_order = Order(
            id=f"{order.id}_period_{start_time.timestamp()}",
            symbol=order.symbol,
            side=order.side,
            quantity=target_size,
            order_type=OrderType.LIMIT,
            limit_price=await self._get_best_price(order.symbol, order.side)
        )
        
        return await self._execute_single_order(sub_order)
    
    async def _get_aggressive_price(self, symbol: str, side: str) -> float:
        """Obtiene precio agresivo para ejecución inmediata"""
        ticker = await self._fetch_ticker(symbol)
        
        if side == 'buy':
            # Precio ligeramente sobre el ask
            return ticker['ask'] * 1.001
        else:
            # Precio ligeramente bajo el bid
            return ticker['bid'] * 0.999
    
    async def _get_best_price(self, symbol: str, side: str) -> float:
        """Obtiene mejor precio actual"""
        ticker = await self._fetch_ticker(symbol)
        return ticker['ask'] if side == 'buy' else ticker['bid']
    
    async def _fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker simulado"""
        # En producción, esto haría una llamada real a la API
        base_price = 100.0
        spread = 0.01
        
        return {
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'last': base_price,
            'volume': random.randint(100000, 1000000)
        }
    
    async def _get_volume_profile(self, symbol: str) -> List[Dict]:
        """Obtiene perfil de volumen intradía"""
        # Simular perfil típico de volumen (U-shaped)
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30)
        
        profile = []
        
        # Volumen alto en apertura
        for i in range(4):  # Primera hora
            profile.append({
                'start_time': market_open + timedelta(minutes=i*15),
                'end_time': market_open + timedelta(minutes=(i+1)*15),
                'expected_volume': 100000 * (1.5 - i*0.1)  # Decrece
            })
        
        # Volumen bajo en medio día
        for i in range(4, 20):  # Medio día
            profile.append({
                'start_time': market_open + timedelta(minutes=i*15),
                'end_time': market_open + timedelta(minutes=(i+1)*15),
                'expected_volume': 50000  # Bajo constante
            })
        
        # Volumen alto en cierre
        for i in range(20, 26):  # Última hora y media
            profile.append({
                'start_time': market_open + timedelta(minutes=i*15),
                'end_time': market_open + timedelta(minutes=(i+1)*15),
                'expected_volume': 100000 * (1 + (i-20)*0.2)  # Incrementa
            })
        
        return profile
    
    def _calculate_optimal_slice_size(self, order: Order, market_conditions: Dict) -> float:
        """Calcula el tamaño óptimo de slice basado en condiciones del mercado"""
        base_percentage = 0.05  # 5% base
        
        # Ajustar por liquidez
        if market_conditions['liquidity'] == 'low':
            base_percentage *= 0.5
        elif market_conditions['liquidity'] == 'high':
            base_percentage *= 1.5
        
        # Ajustar por volatilidad
        if market_conditions['volatility'] > 0.02:
            base_percentage *= 0.7
        
        # Ajustar por urgencia
        if order.time_in_force == 'ioc':
            base_percentage *= 2
        
        return order.quantity * base_percentage

class AdaptiveExecutionEngine:
    """Motor de ejecución adaptativo con ML"""
    
    def __init__(self):
        self.execution_models = {}
        self.performance_history = []
        self.market_impact_model = None
        
    def train_execution_models(self, historical_executions: pd.DataFrame):
        """Entrena modelos de ejecución óptima"""
        # Features: tamaño, spread, volatilidad, volumen, hora del día
        # Target: slippage minimizado
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            features = ['order_size', 'spread', 'volatility', 'volume', 'hour', 'minute']
            target = 'slippage'
            
            # Verificar que las columnas existen
            if all(col in historical_executions.columns for col in features + [target]):
                X = historical_executions[features]
                y = historical_executions[target]
                
                self.market_impact_model = RandomForestRegressor(n_estimators=100)
                self.market_impact_model.fit(X, y)
        except ImportError:
            print("sklearn not available - using default strategies")
    
    def predict_optimal_execution(self, order: Order, market_state: Dict) -> Dict:
        """Predice estrategia de ejecución óptima"""
        if not self.market_impact_model:
            return self._default_strategy(order)
        
        # Preparar features
        features = pd.DataFrame([{
            'order_size': order.quantity,
            'spread': market_state['spread'],
            'volatility': market_state['volatility'],
            'volume': market_state['volume'],
            'hour': datetime.now().hour,
            'minute': datetime.now().minute
        }])
        
        # Predecir slippage para diferentes estrategias
        strategies = {
            'aggressive': {'urgency': 1.0, 'participation': 0.2},
            'passive': {'urgency': 0.3, 'participation': 0.05},
            'balanced': {'urgency': 0.6, 'participation': 0.1}
        }
        
        best_strategy = None
        min_cost = float('inf')
        
        for name, params in strategies.items():
            predicted_slippage = self.market_impact_model.predict(features)[0]
            total_cost = predicted_slippage + params['urgency'] * 0.001  # Costo de oportunidad
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_strategy = name
        
        return {
            'strategy': best_strategy,
            'params': strategies[best_strategy],
            'predicted_slippage': min_cost
        }
    
    def _default_strategy(self, order: Order) -> Dict:
        """Estrategia por defecto sin ML"""
        if order.quantity > 10000:
            return {
                'strategy': 'twap',
                'params': {'duration': 30, 'slices': 10}
            }
        else:
            return {
                'strategy': 'immediate',
                'params': {}
            }

class RealTimeMarketMicrostructure:
    """Análisis en tiempo real de microestructura del mercado"""
    
    def __init__(self):
        self.order_flow_imbalance = {}
        self.toxic_flow_detector = None
        self.quote_stuffing_detector = None
        
    async def analyze_order_flow(self, symbol: str) -> Dict:
        """Analiza el flujo de órdenes en tiempo real"""
        # Obtener datos de nivel 2
        order_book = await self._fetch_level2_data(symbol)
        
        # Calcular imbalance
        bid_volume = sum(level['size'] for level in order_book['bids'][:5])
        ask_volume = sum(level['size'] for level in order_book['asks'][:5])
        
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Detectar agresores
        recent_trades = await self._fetch_recent_trades(symbol)
        buy_aggressor_volume = sum(t['size'] for t in recent_trades if t['side'] == 'buy')
        sell_aggressor_volume = sum(t['size'] for t in recent_trades if t['side'] == 'sell')
        
        total_aggressor_volume = buy_aggressor_volume + sell_aggressor_volume
        buy_pressure = buy_aggressor_volume / total_aggressor_volume if total_aggressor_volume > 0 else 0.5
        
        # Detectar momentum
        price_momentum = self._calculate_price_momentum(recent_trades)
        
        return {
            'order_flow_imbalance': imbalance,
            'buy_pressure': buy_pressure,
            'price_momentum': price_momentum,
            'toxic_flow_probability': self._detect_toxic_flow(recent_trades),
            'execution_recommendation': self._recommend_execution(imbalance, price_momentum)
        }
    
    async def _fetch_level2_data(self, symbol: str) -> Dict:
        """Fetch level 2 data simulado"""
        # Simulación de order book nivel 2
        base_price = 100.0
        
        return {
            'bids': [
                {'price': base_price - 0.01 * (i + 1), 'size': random.randint(100, 1000)}
                for i in range(10)
            ],
            'asks': [
                {'price': base_price + 0.01 * (i + 1), 'size': random.randint(100, 1000)}
                for i in range(10)
            ]
        }
    
    async def _fetch_recent_trades(self, symbol: str) -> List[Dict]:
        """Fetch trades recientes simulados"""
        now = datetime.now()
        trades = []
        
        for i in range(20):
            trades.append({
                'timestamp': now - timedelta(seconds=i*3),
                'price': 100.0 + random.uniform(-0.5, 0.5),
                'size': random.randint(10, 1000),
                'side': random.choice(['buy', 'sell'])
            })
        
        return trades
    
    def _calculate_price_momentum(self, trades: List[Dict]) -> float:
        """Calcula momentum del precio"""
        if len(trades) < 2:
            return 0.0
        
        # Precio promedio primeros 10 trades vs últimos 10
        first_half = trades[:len(trades)//2]
        second_half = trades[len(trades)//2:]
        
        avg_price_first = np.mean([t['price'] for t in first_half])
        avg_price_second = np.mean([t['price'] for t in second_half])
        
        momentum = (avg_price_second - avg_price_first) / avg_price_first
        
        return momentum
    
    def _detect_toxic_flow(self, trades: List[Dict]) -> float:
        """Detecta flujo tóxico (informed traders)"""
        if len(trades) < 10:
            return 0.0
        
        # Características de flujo tóxico:
        # 1. Trades grandes y unidireccionales
        # 2. Ejecución rápida
        # 3. Precio se mueve en dirección del trade
        
        # Calcular métricas
        sizes = [t['size'] for t in trades]
        avg_size = np.mean(sizes)
        size_std = np.std(sizes)
        
        # Trades grandes
        large_trades = [t for t in trades if t['size'] > avg_size + 2*size_std]
        
        if not large_trades:
            return 0.0
        
        # Direccionalidad
        buy_large = sum(1 for t in large_trades if t['side'] == 'buy')
        sell_large = sum(1 for t in large_trades if t['side'] == 'sell')
        directionality = abs(buy_large - sell_large) / len(large_trades)
        
        # Velocidad
        if len(trades) > 1:
            time_span = (trades[0]['timestamp'] - trades[-1]['timestamp']).total_seconds()
            velocity = len(large_trades) / max(time_span, 1)
        else:
            velocity = 0
        
        # Score de toxicidad
        toxicity_score = (directionality * 0.5 + min(velocity/10, 1) * 0.5)
        
        return toxicity_score
    
    def _recommend_execution(self, imbalance: float, momentum: float) -> str:
        """Recomienda estrategia de ejecución basada en microestructura"""
        if abs(imbalance) > 0.3 and momentum * imbalance > 0:
            # Fuerte presión en una dirección
            return "aggressive"
        elif abs(imbalance) < 0.1:
            # Mercado balanceado
            return "passive"
        else:
            return "adaptive"

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar brokers simulados
    broker_connections = {
        'broker1': {'latency': 0.5},
        'broker2': {'latency': 1.0},
        'broker3': {'latency': 0.8}
    }
    
    # Crear router y ejecutor
    router = SmartOrderRouter(broker_connections)
    executor = AdvancedOrderExecutor(router)
    
    # Iniciar ejecutor
    executor.start()
    
    # Crear orden de ejemplo
    order = Order(
        id="TEST001",
        symbol="BTC/USD",
        side="buy",
        quantity=10.0,
        order_type=OrderType.TWAP,
        twap_duration=10  # 10 minutos
    )
    
    # Ejecutar orden
    asyncio.run(executor.execute_order(order))
    
    # Detener ejecutor
    executor.stop()