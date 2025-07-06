# src/agente_momentum.py
from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

# Configuración Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AgenteMomentum:
    def __init__(self):
        self.nombre = "Agente Momentum"
        self.capital = self.get_capital()
        print(f"✅ {self.nombre} iniciado con ${self.capital}")
    
    def get_capital(self):
        """Obtiene capital actual de la BD"""
        try:
            response = supabase.table('bot_status').select("capital").execute()
            return float(response.data[0]['capital']) if response.data else 200.0
        except:
            return 200.0
    
    def calculate_rsi(self, data, period=14):
        """Calcula el RSI (Relative Strength Index)"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            if loss.iloc[-1] == 0:
                return 100 if gain.iloc[-1] > 0 else 50
            
            rs = gain.iloc[-1] / loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50
    
    def calculate_macd(self, data):
        """Calcula MACD (Moving Average Convergence Divergence)"""
        try:
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Detectar cruces
            cross = 'NONE'
            if len(histogram) >= 2:
                if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                    cross = 'BULLISH'
                elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                    cross = 'BEARISH'
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': histogram.iloc[-1],
                'cross': cross
            }
        except:
            return {
                'macd': 0,
                'signal': 0,
                'histogram': 0,
                'cross': 'NONE'
            }
    
    def calculate_momentum_score(self, data):
        """Calcula un score de momentum compuesto - VERSIÓN MEJORADA"""
        if len(data) < 26:
            return 0, {}
        
        try:
            precio_actual = data['Close'].iloc[-1]
            
            # Momentum a diferentes plazos
            mom_1d = (precio_actual - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            mom_5d = (precio_actual - data['Close'].iloc[-5]) / data['Close'].iloc[-5] if len(data) >= 5 else 0
            mom_10d = (precio_actual - data['Close'].iloc[-10]) / data['Close'].iloc[-10] if len(data) >= 10 else 0
            mom_20d = (precio_actual - data['Close'].iloc[-20]) / data['Close'].iloc[-20] if len(data) >= 20 else 0
            
            # RSI
            rsi = self.calculate_rsi(data)
            
            # MACD
            macd_data = self.calculate_macd(data)
            
            # Volatilidad
            volatilidad = data['Close'][-20:].std() / data['Close'][-20:].mean() if len(data) >= 20 else 0.02
            
            # === NUEVO: Análisis de aceleración ===
            # Ver si el momentum está acelerando
            aceleracion = mom_5d - mom_10d if mom_10d != 0 else 0
            
            # === NUEVO: Fuerza de la tendencia ===
            # Contar días positivos vs negativos
            cambios_diarios = data['Close'].pct_change()[-10:]
            dias_positivos = sum(cambios_diarios > 0)
            fuerza_tendencia = dias_positivos / 10
            
            # Calcular score compuesto - AJUSTADO
            score = 0
            
            # === PESOS AJUSTADOS PARA MÁS SEÑALES BUY ===
            
            # Peso por momentum a diferentes plazos (más sensible)
            if mom_1d > 0.005:  # Reducido de 0.01
                score += 0.20  # Aumentado de 0.15
            if mom_5d > 0.01:   # Reducido de 0.02
                score += 0.25  # Aumentado de 0.20
            if mom_10d > 0.02:  # Reducido de 0.03
                score += 0.20  # Aumentado de 0.15
            if mom_20d > 0.05:  # Criterio para tendencia fuerte
                score += 0.15  # Aumentado de 0.10
            
            # Peso por RSI (más balanceado)
            if 45 < rsi < 65:  # Zona neutral con momentum
                score += 0.20  # Aumentado de 0.15
            elif 65 <= rsi < 75:  # Momentum fuerte sin sobrecompra extrema
                score += 0.15  # Nuevo
            elif rsi > 75:  # Sobrecompra
                score += 0.05  # Mantener algo de peso
            elif rsi < 30:  # Sobreventa
                score -= 0.10
            
            # Peso por MACD
            if macd_data['histogram'] > 0: 
                score += 0.15
            if macd_data['cross'] == 'BULLISH': 
                score += 0.15  # Aumentado de 0.10
            elif macd_data['cross'] == 'BEARISH':
                score -= 0.10
            
            # === NUEVO: Bonus por aceleración ===
            if aceleracion > 0.01:  # Momentum acelerando
                score += 0.10
            
            # === NUEVO: Bonus por tendencia consistente ===
            if fuerza_tendencia > 0.7:  # 70% días positivos
                score += 0.10
            
            # Ajuste por volatilidad (menos penalización)
            if volatilidad < 0.02: 
                score *= 1.1  # Reducido de 1.2
            elif volatilidad > 0.05: 
                score *= 0.9  # Reducido de 0.8
            
            return score, {
                'mom_1d': mom_1d,
                'mom_5d': mom_5d,
                'mom_10d': mom_10d,
                'mom_20d': mom_20d,
                'rsi': rsi,
                'macd': macd_data,
                'volatilidad': volatilidad,
                'aceleracion': aceleracion,
                'fuerza_tendencia': fuerza_tendencia
            }
        except Exception as e:
            print(f"  ❌ Error calculando momentum score: {e}")
            return 0, {}
    
    def analyze_symbol(self, symbol):
        """Análisis avanzado de momentum - VERSIÓN MEJORADA"""
        try:
            # Descargar datos
            stock = yf.Ticker(symbol)
            data = stock.history(period="2mo")
            
            if len(data) < 26:
                return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Datos insuficientes'}
            
            precio_actual = data['Close'].iloc[-1]
            volumen_actual = data['Volume'].iloc[-1]
            volumen_promedio = data['Volume'][-20:].mean()
            
            # Calcular score de momentum
            score, indicadores = self.calculate_momentum_score(data)
            
            # Imprimir información de debug
            print(f"  📊 {symbol}: Score momentum = {score:.2f}")
            print(f"  📈 Momentum: 1d={indicadores.get('mom_1d', 0)*100:+.1f}%, "
                  f"5d={indicadores.get('mom_5d', 0)*100:+.1f}%, "
                  f"20d={indicadores.get('mom_20d', 0)*100:+.1f}%")
            print(f"  📉 RSI: {indicadores.get('rsi', 50):.0f}, "
                  f"MACD: {indicadores.get('macd', {}).get('cross', 'NONE')}")
            if indicadores.get('aceleracion', 0) > 0:
                print(f"  🚀 Momentum acelerando: {indicadores['aceleracion']*100:+.1f}%")
            
            # === ESTRATEGIAS DE MOMENTUM MEJORADAS ===
            
            # 1. Momentum ultra-fuerte (umbral reducido)
            if score > 0.65 and indicadores.get('mom_5d', 0) > 0.02:  # Reducido de 0.7 y 0.03
                return {
                    'action': 'BUY',
                    'confidence': min(0.85, score),
                    'price': precio_actual,
                    'reason': f'Momentum fuerte: Score {score:.2f}, +{indicadores["mom_5d"]*100:.1f}% en 5d'
                }
            
            # 2. Breakout momentum (más sensible)
            elif indicadores.get('mom_1d', 0) > 0.015 and volumen_actual > volumen_promedio * 1.3:  # Reducido de 0.02 y 1.5
                return {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'price': precio_actual,
                    'reason': f'Breakout: +{indicadores["mom_1d"]*100:.1f}% con volumen alto'
                }
            
            # 3. MACD crossover
            elif indicadores.get('macd', {}).get('cross') == 'BULLISH' and indicadores.get('mom_5d', 0) > -0.01:  # Más tolerante
                return {
                    'action': 'BUY',
                    'confidence': 0.70,
                    'price': precio_actual,
                    'reason': 'MACD bullish cross + momentum positivo'
                }
            
            # 4. Momentum sostenido (criterios más flexibles)
            elif all([
                indicadores.get('mom_5d', 0) > 0.01,   # Reducido de 0.02
                indicadores.get('mom_10d', 0) > 0.02,  # Reducido de 0.03
                indicadores.get('mom_20d', 0) > 0.03   # Reducido de 0.05
            ]):
                return {
                    'action': 'BUY',
                    'confidence': 0.65,
                    'price': precio_actual,
                    'reason': f'Tendencia sostenida: +{indicadores["mom_20d"]*100:.1f}% en 20d'
                }
            
            # === NUEVO: Momentum inicial ===
            elif score > 0.5 and indicadores.get('aceleracion', 0) > 0.01:
                return {
                    'action': 'BUY',
                    'confidence': 0.60,
                    'price': precio_actual,
                    'reason': f'Momentum acelerando: Score {score:.2f}, aceleración +{indicadores["aceleracion"]*100:.1f}%'
                }
            
            # === NUEVO: Recuperación desde caída ===
            elif indicadores.get('mom_5d', 0) > 0.03 and indicadores.get('mom_20d', 0) < -0.05:
                return {
                    'action': 'BUY',
                    'confidence': 0.55,
                    'price': precio_actual,
                    'reason': f'Recuperación rápida: +{indicadores["mom_5d"]*100:.1f}% tras caída'
                }
            
            # 5. Momentum negativo fuerte (mantener conservador en ventas)
            elif score < 0.25 and indicadores.get('mom_5d', 0) < -0.04:  # Más estricto
                return {
                    'action': 'SELL',
                    'confidence': min(0.75, 1 - score),
                    'price': precio_actual,
                    'reason': f'Momentum negativo: Score {score:.2f}, {indicadores["mom_5d"]*100:.1f}% en 5d'
                }
            
            # 6. MACD bearish
            elif indicadores.get('macd', {}).get('cross') == 'BEARISH' and indicadores.get('mom_5d', 0) < -0.01:
                return {
                    'action': 'SELL',
                    'confidence': 0.65,  # Reducido de 0.70
                    'price': precio_actual,
                    'reason': 'MACD bearish cross + momentum negativo'
                }
            
            # 7. Pérdida de momentum en sobrecompra extrema
            elif indicadores.get('rsi', 50) > 80 and indicadores.get('mom_1d', 0) < -0.015:  # Más estricto
                return {
                    'action': 'SELL',
                    'confidence': 0.60,  # Reducido de 0.65
                    'price': precio_actual,
                    'reason': f'Pérdida momentum en sobrecompra extrema: RSI {indicadores["rsi"]:.0f}'
                }
            
            # Sin señal clara
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': precio_actual,
                'reason': f'Sin señal clara (Score: {score:.2f})'
            }
            
        except Exception as e:
            print(f"  ❌ Error analizando {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': f'Error en análisis: {str(e)}'
            }
    
    def execute_trade(self, symbol, signal):
        """Registra trade en la BD"""
        if signal['action'] in ['BUY', 'SELL']:
            try:
                # Calcular cantidad
                cantidad = (self.capital * 0.20) / signal['price']
                
                # Guardar en BD
                trade = {
                    'symbol': symbol,
                    'action': signal['action'],
                    'quantity': cantidad,
                    'price': signal['price'],
                    'agent_confidence': signal['confidence'],
                    'agent_name': self.nombre,
                    'strategy_reason': signal['reason']
                }
                
                supabase.table('trades').insert(trade).execute()
                
                emoji = "🚀" if signal['action'] == 'BUY' else "💥"
                print(f"{emoji} {signal['action']} (Momentum): {cantidad:.2f} {symbol} @ ${signal['price']:.2f}")
                print(f"   Razón: {signal['reason']}")
                print(f"   Confianza: {signal['confidence']*100:.0f}%")
            except Exception as e:
                print(f"❌ Error ejecutando trade: {e}")
    
    def run_cycle(self, symbols):
        """Ejecuta un ciclo de análisis"""
        print(f"\n⏰ Ciclo Momentum - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        encontro_algo = False
        for symbol in symbols:
            print(f"\nAnalizando {symbol}...")
            signal = self.analyze_symbol(symbol)
            
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                encontro_algo = True
        
        if not encontro_algo:
            print("\n🔍 No encontré oportunidades de momentum claras")
        
        print("\n✅ Ciclo completado")


if __name__ == "__main__":
    print("🚀 MOMENTUM TRADING AGENT - VERSIÓN MEJORADA")
    print("Detectando tendencias fuertes y aceleración de precios")
    print("="*50)
    
    # Crear agente
    agente = AgenteMomentum()
    
    # Símbolos a analizar
    from config import SYMBOLS
    
    # Ejecutar análisis
    agente.run_cycle(SYMBOLS)