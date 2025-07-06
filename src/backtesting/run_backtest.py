# src/run_backtest.py - Versión corregida con fix para RuntimeError

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Importar los agentes
from strategies.agente_momentum import AgenteMomentum
from strategies.agente_mean_reversion import AgenteMeanReversion
from strategies.agente_pattern_recognition import AgentePatternRecognition
from strategies.agente_volume_momentum import AgenteVolumeMomentum
from strategies.agente_sentiment import AgenteSentiment

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        """Motor de backtesting profesional"""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {symbol: {'qty': float, 'entry_price': float, 'entry_date': str}}
        self.trades_history = []
        self.equity_history = []
        self.daily_returns = []
        
        # Métricas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Configuración de costos
        self.commission = 0.001  # 0.1% por trade
        self.slippage = 0.0005   # 0.05% slippage
        
        # Risk Management
        self.max_position_size = 0.20  # 20% máximo por posición
        self.stop_loss = 0.05         # 5% stop loss
        self.take_profit = 0.15       # 15% take profit
        
        print(f"🧪 BacktestEngine iniciado con ${initial_capital:,.2f}")
        
        # Inicializar agentes SIN conexión a Supabase
        self.agentes = self._init_agentes_backtest()
    
    def _init_agentes_backtest(self):
        """Inicializa agentes para backtest sin BD"""
        agentes = []
        
        try:
            # Crear versiones simplificadas de los agentes
            class MockAgent:
                def __init__(self, agent_class, name):
                    self.nombre = name
                    self.agent_class = agent_class
                    
                def analyze_symbol(self, symbol, data=None):
                    """Análisis simplificado para backtest"""
                    try:
                        # Crear instancia temporal del agente
                        if self.agent_class == AgenteMomentum:
                            return self._analyze_momentum(symbol, data)
                        elif self.agent_class == AgenteMeanReversion:
                            return self._analyze_mean_reversion(symbol, data)
                        elif self.agent_class == AgentePatternRecognition:
                            return self._analyze_patterns(symbol, data)
                        elif self.agent_class == AgenteVolumeMomentum:
                            return self._analyze_volume(symbol, data)
                        elif self.agent_class == AgenteSentiment:
                            return self._analyze_sentiment(symbol, data)
                    except:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Error en análisis'}
                
                def _analyze_momentum(self, symbol, data):
                    """Análisis básico de momentum"""
                    if len(data) < 20:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
                    
                    # Calcular momentum básico
                    price = data['Close'].iloc[-1]
                    sma_5 = data['Close'].rolling(5).mean().iloc[-1]
                    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                    
                    # RSI básico
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 0
                    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                    
                    # Lógica de momentum
                    if sma_5 > sma_20 * 1.02 and rsi > 55 and rsi < 75:
                        return {
                            'action': 'BUY',
                            'confidence': 0.7,
                            'reason': f'Momentum alcista: SMA5 > SMA20, RSI {rsi:.0f}'
                        }
                    elif sma_5 < sma_20 * 0.98 and rsi < 45:
                        return {
                            'action': 'SELL',
                            'confidence': 0.7,
                            'reason': f'Momentum bajista: SMA5 < SMA20, RSI {rsi:.0f}'
                        }
                    
                    return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Sin momentum claro'}
                
                def _analyze_mean_reversion(self, symbol, data):
                    """Análisis básico de reversión a la media"""
                    if len(data) < 20:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
                    
                    price = data['Close'].iloc[-1]
                    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                    std_20 = data['Close'].rolling(20).std().iloc[-1]
                    
                    # Bollinger Bands
                    upper_bb = sma_20 + (std_20 * 2)
                    lower_bb = sma_20 - (std_20 * 2)
                    
                    # RSI
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 0
                    rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                    
                    # Lógica de reversión
                    if price <= lower_bb and rsi < 30:
                        return {
                            'action': 'BUY',
                            'confidence': 0.75,
                            'reason': f'Sobreventa extrema: BB inferior, RSI {rsi:.0f}'
                        }
                    elif price >= upper_bb and rsi > 70:
                        return {
                            'action': 'SELL',
                            'confidence': 0.75,
                            'reason': f'Sobrecompra extrema: BB superior, RSI {rsi:.0f}'
                        }
                    
                    return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'En rango normal'}
                
                def _analyze_patterns(self, symbol, data):
                    """Análisis básico de patrones"""
                    if len(data) < 30:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
                    
                    # Detectar breakouts simples
                    price = data['Close'].iloc[-1]
                    high_20 = data['High'].rolling(20).max().iloc[-2]  # Excluir día actual
                    low_20 = data['Low'].rolling(20).min().iloc[-2]
                    volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
                    volume_current = data['Volume'].iloc[-1]
                    
                    # Breakout alcista
                    if price > high_20 and volume_current > volume_avg * 1.5:
                        return {
                            'action': 'BUY',
                            'confidence': 0.8,
                            'reason': 'Breakout alcista con volumen'
                        }
                    # Breakdown bajista
                    elif price < low_20 and volume_current > volume_avg * 1.5:
                        return {
                            'action': 'SELL',
                            'confidence': 0.8,
                            'reason': 'Breakdown bajista con volumen'
                        }
                    
                    return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Sin patrones claros'}
                
                def _analyze_volume(self, symbol, data):
                    """Análisis básico de volumen"""
                    if len(data) < 20:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
                    
                    price = data['Close'].iloc[-1]
                    volume_current = data['Volume'].iloc[-1]
                    volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
                    price_change = (price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                    
                    # Volumen explosivo
                    if volume_current > volume_avg * 2 and abs(price_change) > 0.02:
                        if price_change > 0:
                            return {
                                'action': 'BUY',
                                'confidence': 0.75,
                                'reason': f'Volumen explosivo alcista: {volume_current/volume_avg:.1f}x'
                            }
                        else:
                            return {
                                'action': 'SELL',
                                'confidence': 0.75,
                                'reason': f'Volumen explosivo bajista: {volume_current/volume_avg:.1f}x'
                            }
                    
                    return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Volumen normal'}
                
                def _analyze_sentiment(self, symbol, data):
                    """Análisis básico de sentiment (simplificado)"""
                    if len(data) < 10:
                        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
                    
                    # Sentiment basado en price action
                    returns = data['Close'].pct_change().dropna()
                    recent_returns = returns.tail(5)
                    avg_return = recent_returns.mean()
                    
                    if avg_return > 0.01:  # Promedio positivo fuerte
                        return {
                            'action': 'BUY',
                            'confidence': 0.6,
                            'reason': f'Sentiment positivo: +{avg_return*100:.1f}% promedio'
                        }
                    elif avg_return < -0.01:  # Promedio negativo fuerte
                        return {
                            'action': 'SELL',
                            'confidence': 0.6,
                            'reason': f'Sentiment negativo: {avg_return*100:.1f}% promedio'
                        }
                    
                    return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment neutral'}
            
            # Crear agentes mock
            agentes.append(MockAgent(AgenteMomentum, "Momentum Agent"))
            agentes.append(MockAgent(AgenteMeanReversion, "Mean Reversion Agent"))
            agentes.append(MockAgent(AgentePatternRecognition, "Pattern Agent"))
            agentes.append(MockAgent(AgenteVolumeMomentum, "Volume Agent"))
            agentes.append(MockAgent(AgenteSentiment, "Sentiment Agent"))
            
            print(f"✅ {len(agentes)} agentes inicializados para backtest")
            return agentes
            
        except Exception as e:
            print(f"⚠️ Error inicializando agentes: {e}")
            return []
    
    def get_consensus(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Obtiene consenso de todos los agentes"""
        if len(self.agentes) == 0:
            return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Sin agentes disponibles'}
        
        votos = []
        for agente in self.agentes:
            try:
                voto = agente.analyze_symbol(symbol, data)
                votos.append(voto)
            except Exception as e:
                # En caso de error, voto neutral
                votos.append({'action': 'HOLD', 'confidence': 0.5, 'reason': f'Error: {e}'})
        
        # Calcular consenso simple
        buy_votes = sum(1 for v in votos if v['action'] == 'BUY')
        sell_votes = sum(1 for v in votos if v['action'] == 'SELL')
        hold_votes = sum(1 for v in votos if v['action'] == 'HOLD')
        
        # Confianza promedio por acción
        buy_confidence = np.mean([v['confidence'] for v in votos if v['action'] == 'BUY']) if buy_votes > 0 else 0
        sell_confidence = np.mean([v['confidence'] for v in votos if v['action'] == 'SELL']) if sell_votes > 0 else 0
        
        # Decisión por mayoría
        if buy_votes >= 3 and buy_confidence > 0.6:
            return {
                'action': 'BUY',
                'confidence': buy_confidence,
                'reason': f'Consenso BUY: {buy_votes}/5 agentes'
            }
        elif sell_votes >= 3 and sell_confidence > 0.6:
            return {
                'action': 'SELL',
                'confidence': sell_confidence,
                'reason': f'Consenso SELL: {sell_votes}/5 agentes'
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': f'Sin consenso: B{buy_votes}/S{sell_votes}/H{hold_votes}'
            }
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calcula tamaño de posición basado en Kelly Criterion"""
        # Kelly Criterion simplificado
        win_rate = 0.6  # Asumido
        avg_win = 0.15  # 15% ganancia promedio
        avg_loss = 0.05  # 5% pérdida promedio
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Ajustar por confianza
        kelly_adjusted = kelly * confidence * 0.5  # Factor conservador
        
        # Límites
        return max(0.05, min(self.max_position_size, kelly_adjusted))
    
    def execute_trade(self, symbol: str, action: str, price: float, date: str, confidence: float, reason: str):
        """Ejecuta un trade en el backtest"""
        try:
            if action == 'BUY' and symbol not in self.positions:
                # Calcular cantidad
                position_size = self.calculate_position_size(symbol, confidence)
                trade_value = self.capital * position_size
                quantity = trade_value / price
                
                # Aplicar costos
                commission_cost = trade_value * self.commission
                slippage_cost = trade_value * self.slippage
                total_cost = trade_value + commission_cost + slippage_cost
                
                if total_cost <= self.capital:
                    # Ejecutar compra
                    self.positions[symbol] = {
                        'qty': quantity,
                        'entry_price': price * (1 + self.slippage),
                        'entry_date': date,
                        'position_size': position_size
                    }
                    
                    self.capital -= total_cost
                    
                    self.trades_history.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'value': trade_value,
                        'commission': commission_cost,
                        'reason': reason,
                        'confidence': confidence
                    })
                    
                    self.total_trades += 1
                    return True
            
            elif action == 'SELL' and symbol in self.positions:
                # Vender posición existente
                position = self.positions[symbol]
                quantity = position['qty']
                entry_price = position['entry_price']
                
                # Calcular valor de venta
                sell_value = quantity * price * (1 - self.slippage)
                commission_cost = sell_value * self.commission
                net_proceeds = sell_value - commission_cost
                
                # Calcular P&L
                entry_value = quantity * entry_price
                pnl = net_proceeds - entry_value
                pnl_pct = (pnl / entry_value) * 100
                
                # Actualizar capital
                self.capital += net_proceeds
                
                # Registrar trade
                self.trades_history.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'value': sell_value,
                    'commission': commission_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_date': position['entry_date'],
                    'entry_price': entry_price,
                    'reason': reason,
                    'confidence': confidence
                })
                
                # Actualizar estadísticas
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Remover posición
                del self.positions[symbol]
                
                self.total_trades += 1
                return True
            
            return False
            
        except Exception as e:
            print(f"Error ejecutando trade {action} {symbol}: {e}")
            return False
    
    def check_stop_loss_take_profit(self, date: str, current_prices: Dict[str, float]):
        """Verifica stop loss y take profit para todas las posiciones"""
        # FIX: Crear una copia de las claves para iterar de forma segura
        positions_to_check = list(self.positions.keys())
        
        for symbol in positions_to_check:
            if symbol not in current_prices:
                continue
                
            position = self.positions[symbol]
            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            
            # Calcular cambio porcentual
            price_change = (current_price - entry_price) / entry_price
            
            # Stop Loss
            if price_change <= -self.stop_loss:
                self.execute_trade(
                    symbol, 'SELL', current_price, date, 1.0,
                    f"Stop Loss: {price_change*100:.1f}%"
                )
            
            # Take Profit
            elif price_change >= self.take_profit:
                self.execute_trade(
                    symbol, 'SELL', current_price, date, 1.0,
                    f"Take Profit: {price_change*100:.1f}%"
                )
    
    def update_equity(self, date: str, current_prices: Dict[str, float]):
        """Actualiza el equity total"""
        # Calcular valor de posiciones
        positions_value = 0
        # FIX: Crear una copia de los items para iterar de forma segura
        position_items = list(self.positions.items())
        
        for symbol, position in position_items:
            if symbol in current_prices:
                current_value = position['qty'] * current_prices[symbol]
                positions_value += current_value
        
        total_equity = self.capital + positions_value
        
        self.equity_history.append({
            'date': date,
            'cash': self.capital,
            'positions_value': positions_value,
            'total_equity': total_equity
        })
        
        # Calcular retorno diario
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]['total_equity']
            daily_return = (total_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def execute_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Ejecuta el backtest completo"""
        print(f"\n🚀 INICIANDO BACKTEST")
        print(f"📅 Período: {start_date} a {end_date}")
        print(f"💰 Capital inicial: ${self.initial_capital:,.2f}")
        print(f"📊 Símbolos: {', '.join(symbols)}")
        print("="*60)
        
        # Descargar datos
        print("📥 Descargando datos históricos...")
        all_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if len(data) > 0:
                    all_data[symbol] = data
                    print(f"  ✅ {symbol}: {len(data)} días")
                else:
                    print(f"  ❌ {symbol}: Sin datos")
            except Exception as e:
                print(f"  ❌ {symbol}: Error - {e}")
        
        if not all_data:
            print("❌ No se pudieron descargar datos")
            return {}
        
        # Obtener fechas comunes
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = []
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            # Verificar si hay datos para al menos un símbolo
            if any(date_str in data.index.strftime('%Y-%m-%d') for data in all_data.values()):
                trading_days.append(date_str)
        
        print(f"\n📈 Ejecutando backtest: {len(trading_days)} días de trading")
        
        # Loop principal del backtest
        for i, current_date in enumerate(trading_days):
            if i % 50 == 0:
                print(f"  📅 Procesando: {current_date} ({i+1}/{len(trading_days)})")
            
            try:
                # Obtener precios del día
                current_prices = {}
                for symbol, data in all_data.items():
                    try:
                        date_index = pd.to_datetime(current_date).date()
                        day_data = data[data.index.date == date_index]
                        if len(day_data) > 0:
                            current_prices[symbol] = day_data['Close'].iloc[0]
                    except:
                        continue
                
                if not current_prices:
                    continue
                
                # Verificar stop loss y take profit
                self.check_stop_loss_take_profit(current_date, current_prices)
                
                # Analizar señales de los agentes (solo los primeros 20 días para ahorrar tiempo)
                if i >= 20:
                    for symbol, data in all_data.items():
                        if symbol not in current_prices:
                            continue
                        
                        try:
                            # Obtener datos hasta la fecha actual
                            date_index = pd.to_datetime(current_date).date()
                            historical_data = data[data.index.date <= date_index].tail(50)
                            
                            if len(historical_data) < 20:
                                continue
                            
                            # Obtener consenso de agentes
                            consensus = self.get_consensus(symbol, historical_data)
                            
                            if consensus['action'] in ['BUY', 'SELL'] and consensus['confidence'] > 0.65:
                                price = current_prices[symbol]
                                self.execute_trade(
                                    symbol, consensus['action'], price, 
                                    current_date, consensus['confidence'], consensus['reason']
                                )
                        except Exception as e:
                            continue
                
                # Actualizar equity
                self.update_equity(current_date, current_prices)
                
            except Exception as e:
                print(f"  ⚠️ Error en {current_date}: {e}")
                continue
        
        # Cerrar posiciones abiertas al final
        print("\n💼 Cerrando posiciones abiertas...")
        final_prices = {}
        for symbol, data in all_data.items():
            if len(data) > 0:
                final_prices[symbol] = data['Close'].iloc[-1]
        
        # FIX: Crear copia de las claves para iterar de forma segura
        open_positions = list(self.positions.keys())
        for symbol in open_positions:
            if symbol in final_prices:
                self.execute_trade(
                    symbol, 'SELL', final_prices[symbol], 
                    trading_days[-1], 1.0, "Cierre final"
                )
        
        # Calcular métricas finales
        print("\n📊 Calculando métricas...")
        metrics = self.calculate_metrics()
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calcula todas las métricas de performance"""
        if not self.equity_history:
            return {}
        
        # Equity final
        final_equity = self.equity_history[-1]['total_equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Métricas básicas
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        
        # Calcular drawdown
        equity_series = pd.Series([eq['total_equity'] for eq in self.equity_history])
        rolling_max = equity_series.expanding().max()
        drawdown_series = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown_series.min())
        
        # Sharpe ratio
        if len(self.daily_returns) > 1:
            returns_std = np.std(self.daily_returns)
            avg_return = np.mean(self.daily_returns)
            sharpe_ratio = (avg_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        winning_trades_data = [t for t in self.trades_history if t.get('pnl', 0) > 0]
        losing_trades_data = [t for t in self.trades_history if t.get('pnl', 0) < 0]
        
        total_wins = sum(t.get('pnl', 0) for t in winning_trades_data)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades_data))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'avg_win': total_wins / max(1, self.winning_trades),
            'avg_loss': total_losses / max(1, self.losing_trades),
            'equity_history': self.equity_history,
            'trades_history': self.trades_history,
            'daily_returns': self.daily_returns
        }
    
    def print_metrics(self, metrics: Dict):
        """Imprime métricas de forma bonita"""
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL BACKTEST")
        print("="*60)
        
        print(f"\n💰 RENDIMIENTO:")
        print(f"  • Capital inicial: ${metrics['initial_capital']:,.2f}")
        print(f"  • Capital final: ${metrics['final_equity']:,.2f}")
        print(f"  • Retorno total: {metrics['total_return_pct']:+.2f}%")
        
        print(f"\n📈 TRADES:")
        print(f"  • Total de trades: {metrics['total_trades']}")
        print(f"  • Trades ganadores: {metrics['winning_trades']}")
        print(f"  • Trades perdedores: {metrics['losing_trades']}")
        print(f"  • Win Rate: {metrics['win_rate_pct']:.1f}%")
        
        print(f"\n📊 MÉTRICAS DE RIESGO:")
        print(f"  • Máximo Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  • Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\n💸 P&L:")
        print(f"  • Ganancias totales: ${metrics['total_wins']:,.2f}")
        print(f"  • Pérdidas totales: ${metrics['total_losses']:,.2f}")
        print(f"  • Ganancia promedio: ${metrics['avg_win']:,.2f}")
        print(f"  • Pérdida promedio: ${metrics['avg_loss']:,.2f}")
        
        # Evaluación
        print(f"\n🎯 EVALUACIÓN:")
        if metrics['total_return_pct'] > 20:
            print("  🟢 Excelente performance")
        elif metrics['total_return_pct'] > 10:
            print("  🟢 Buena performance")
        elif metrics['total_return_pct'] > 0:
            print("  🟡 Performance positiva")
        else:
            print("  🔴 Performance negativa")
        
        if metrics['win_rate_pct'] > 60:
            print("  🟢 Win rate excelente")
        elif metrics['win_rate_pct'] > 50:
            print("  🟡 Win rate aceptable")
        else:
            print("  🔴 Win rate bajo")
        
        if metrics['max_drawdown_pct'] < 10:
            print("  🟢 Drawdown controlado")
        elif metrics['max_drawdown_pct'] < 20:
            print("  🟡 Drawdown moderado")
        else:
            print("  🔴 Drawdown alto")

def main():
    """Función principal del backtest"""
    # Configuración
    symbols = ['NVDA', 'PLTR', 'SOFI', 'NET', 'AMD']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    initial_capital = 10000
    
    print("🧪 BACKTEST SISTEMA MULTI-AGENTE")
    print("="*60)
    
    # Crear y ejecutar backtest
    backtest = BacktestEngine(initial_capital)
    metrics = backtest.execute_backtest(symbols, start_date, end_date)
    
    if metrics:
        backtest.print_metrics(metrics)
        
        # Graficar resultados
        print("\n📈 Generando gráficos...")
        
        try:
            # Equity Curve
            equity_data = pd.DataFrame(metrics['equity_history'])
            equity_data['date'] = pd.to_datetime(equity_data['date'])
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Equity Curve
            ax1.plot(equity_data['date'], equity_data['total_equity'], 'b-', linewidth=2)
            ax1.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            equity_series = equity_data['total_equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            ax2.fill_between(equity_data['date'], drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # 3. Returns Distribution
            if metrics['daily_returns']:
                daily_returns_pct = [r * 100 for r in metrics['daily_returns']]
                ax3.hist(daily_returns_pct, bins=50, alpha=0.7, color='green')
                ax3.axvline(x=np.mean(daily_returns_pct), color='red', linestyle='--')
                ax3.set_title('Daily Returns Distribution')
                ax3.set_xlabel('Daily Return (%)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
            
            # 4. Trade Analysis
            if metrics['trades_history']:
                trades_df = pd.DataFrame(metrics['trades_history'])
                completed_trades = trades_df[trades_df['action'] == 'SELL']
                if len(completed_trades) > 0:
                    pnl_values = completed_trades['pnl_pct'].tolist()
                    colors = ['green' if x > 0 else 'red' for x in pnl_values]
                    ax4.bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.7)
                    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax4.set_title('Individual Trade P&L')
                    ax4.set_xlabel('Trade Number')
                    ax4.set_ylabel('P&L (%)')
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Gráficos guardados como 'backtest_results.png'")
            
        except Exception as e:
            print(f"⚠️ Error generando gráficos: {e}")
    
    return metrics

if __name__ == "__main__":
    metrics = main()
    
    print("\n✅ Backtest completado")
    print("📊 Revisa los resultados y gráficos generados")
    
    input("\nPresiona Enter para salir...")