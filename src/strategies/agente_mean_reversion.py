# src/agente_mean_reversion.py
from utils.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

# Configuración Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AgenteMeanReversion:
    def __init__(self):
        self.nombre = "Agente Mean Reversion"
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
        """Calcula RSI de forma robusta"""
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
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calcula Bandas de Bollinger"""
        try:
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band.iloc[-1],
                'middle': sma.iloc[-1],
                'lower': lower_band.iloc[-1],
                'bandwidth': ((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]) * 100
            }
        except:
            return None
    
    def calculate_stochastic(self, data, period=14):
        """Calcula Stochastic Oscillator"""
        try:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            
            k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            k_percent = k_percent.fillna(50)
            
            return k_percent.iloc[-1]
        except:
            return 50
    
    def analyze_symbol(self, symbol):
        """Busca oportunidades de reversión a la media - VERSIÓN BALANCEADA"""
        try:
            # Descargar datos
            stock = yf.Ticker(symbol)
            data = stock.history(period="2mo")
            
            if len(data) < 20:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'Datos insuficientes'
                }
            
            # Calcular indicadores
            precio_actual = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else sma_20
            
            # RSI
            rsi = self.calculate_rsi(data)
            
            # Bollinger Bands
            bb = self.calculate_bollinger_bands(data)
            
            # Stochastic
            stoch = self.calculate_stochastic(data)
            
            # Distancia desde la media
            distancia_sma20 = ((precio_actual - sma_20) / sma_20) * 100
            distancia_sma50 = ((precio_actual - sma_50) / sma_50) * 100
            
            # Posición en Bollinger Bands
            bb_position = 0
            if bb:
                bb_range = bb['upper'] - bb['lower']
                if bb_range > 0:
                    bb_position = (precio_actual - bb['lower']) / bb_range
            
            # === NUEVO: Análisis de tendencia para contexto ===
            # Verificar si estamos en tendencia alcista fuerte
            tendencia_alcista = sma_20 > sma_50 if len(data) >= 50 else False
            momentum_positivo = data['Close'].iloc[-5:].mean() > data['Close'].iloc[-10:-5].mean()
            
            # === NUEVO: Ajuste dinámico de umbrales según tendencia ===
            # En tendencia alcista, ser más tolerante con RSI alto
            rsi_umbral_venta = 75 if tendencia_alcista else 70
            rsi_umbral_venta_extrema = 80 if tendencia_alcista else 75
            distancia_umbral_venta = 10 if tendencia_alcista else 7
            
            # Imprimir información
            print(f"  📊 {symbol}: Precio=${precio_actual:.2f}, SMA20=${sma_20:.2f}")
            print(f"  📏 Distancia de media: {distancia_sma20:.2f}%")
            print(f"  📈 RSI: {rsi:.2f}, Stoch: {stoch:.2f}")
            if bb:
                print(f"  📊 BB Position: {bb_position:.2f}, Bandwidth: {bb['bandwidth']:.1f}%")
            if tendencia_alcista:
                print(f"  📈 Tendencia alcista detectada - umbrales ajustados")
            
            # ESTRATEGIAS DE MEAN REVERSION - BALANCEADAS
            
            # 1. Sobreventa extrema (mantener agresivo para compras)
            if distancia_sma20 < -5 and rsi < 30 and stoch < 20:
                return {
                    'action': 'BUY',
                    'confidence': 0.85,
                    'price': precio_actual,
                    'reason': f'Sobreventa extrema: {distancia_sma20:.1f}% bajo SMA, RSI={rsi:.0f}'
                }
            
            # 2. Toque de banda inferior de Bollinger
            elif bb and precio_actual <= bb['lower'] * 1.01 and rsi < 40:
                return {
                    'action': 'BUY',
                    'confidence': 0.75,
                    'price': precio_actual,
                    'reason': f'Rebote en Bollinger inferior, RSI={rsi:.0f}'
                }
            
            # 3. Sobreventa moderada con reversión iniciando
            elif distancia_sma20 < -3 and rsi < 35 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                return {
                    'action': 'BUY',
                    'confidence': 0.70,
                    'price': precio_actual,
                    'reason': f'Reversión desde sobreventa: RSI={rsi:.0f}'
                }
            
            # 4. Divergencia alcista
            elif rsi < 30 and data['Close'].iloc[-1] > data['Low'].iloc[-5:].min():
                return {
                    'action': 'BUY',
                    'confidence': 0.65,
                    'price': precio_actual,
                    'reason': 'Divergencia alcista: precio hace mínimo más alto con RSI bajo'
                }
            
            # === VENTAS - UMBRALES MÁS CONSERVADORES ===
            
            # 5. Sobrecompra extrema (umbral más alto)
            elif distancia_sma20 > distancia_umbral_venta and rsi > rsi_umbral_venta_extrema and stoch > 85:
                # Reducir confianza si hay momentum positivo fuerte
                confianza = 0.75 if momentum_positivo else 0.85
                return {
                    'action': 'SELL',
                    'confidence': confianza,
                    'price': precio_actual,
                    'reason': f'Sobrecompra extrema: +{distancia_sma20:.1f}% sobre SMA, RSI={rsi:.0f}'
                }
            
            # 6. Toque de banda superior de Bollinger (más conservador)
            elif bb and precio_actual >= bb['upper'] * 0.99 and rsi > rsi_umbral_venta:
                confianza = 0.65 if tendencia_alcista else 0.75
                return {
                    'action': 'SELL',
                    'confidence': confianza,
                    'price': precio_actual,
                    'reason': f'Rechazo en Bollinger superior, RSI={rsi:.0f}'
                }
            
            # 7. Sobrecompra moderada con reversión (más selectivo)
            elif distancia_sma20 > 8 and rsi > rsi_umbral_venta and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                # Solo si hay confirmación de reversión
                if data['Close'].iloc[-1] < data['High'].iloc[-2] * 0.98:
                    return {
                        'action': 'SELL',
                        'confidence': 0.65,
                        'price': precio_actual,
                        'reason': f'Reversión confirmada desde sobrecompra: +{distancia_sma20:.1f}% sobre SMA'
                    }
            
            # 8. Mean reversion clásico (más conservador)
            elif abs(distancia_sma20) > 12:  # Aumentado de 10 a 12
                if distancia_sma20 > 0 and not tendencia_alcista:
                    return {
                        'action': 'SELL',
                        'confidence': 0.55,  # Reducida de 0.60
                        'price': precio_actual,
                        'reason': f'Desviación extrema: +{distancia_sma20:.1f}% sobre media'
                    }
                elif distancia_sma20 < 0:
                    return {
                        'action': 'BUY',
                        'confidence': 0.60,
                        'price': precio_actual,
                        'reason': f'Desviación extrema: {distancia_sma20:.1f}% bajo media'
                    }
            
            # === NUEVO: Oportunidades en consolidación ===
            # Si el precio está consolidando cerca de la media
            elif abs(distancia_sma20) < 2 and bb and bb['bandwidth'] < 10:
                # Esperar ruptura
                if data['Close'].iloc[-1] > data['Close'].iloc[-2] * 1.01:
                    return {
                        'action': 'BUY',
                        'confidence': 0.55,
                        'price': precio_actual,
                        'reason': 'Ruptura alcista desde consolidación'
                    }
                elif data['Close'].iloc[-1] < data['Close'].iloc[-2] * 0.99:
                    return {
                        'action': 'SELL',
                        'confidence': 0.55,
                        'price': precio_actual,
                        'reason': 'Ruptura bajista desde consolidación'
                    }
            
            # Sin señal clara
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': precio_actual,
                'reason': 'Sin condiciones de reversión claras'
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
                
                emoji = "📈" if signal['action'] == 'BUY' else "📉"
                print(f"{emoji} {signal['action']} (Mean Rev): {cantidad:.2f} {symbol} @ ${signal['price']:.2f}")
                print(f"   Razón: {signal['reason']}")
                print(f"   Confianza: {signal['confidence']*100:.0f}%")
            except Exception as e:
                print(f"❌ Error ejecutando trade: {e}")
    
    def run_cycle(self, symbols):
        """Ejecuta un ciclo de análisis"""
        print(f"\n⏰ Ciclo Mean Reversion - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        encontro_algo = False
        for symbol in symbols:
            print(f"\nAnalizando {symbol}...")
            signal = self.analyze_symbol(symbol)
            
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                encontro_algo = True
        
        if not encontro_algo:
            print("\n🔍 No encontré oportunidades de reversión a la media")
        
        print("\n✅ Ciclo completado")


if __name__ == "__main__":
    print("📊 MEAN REVERSION AGENT - VERSIÓN BALANCEADA")
    print("Detectando extremos y reversiones con ajuste por tendencia")
    print("="*50)
    
    # Crear agente
    agente = AgenteMeanReversion()
    
    # Símbolos a analizar
    from utils.config import SYMBOLS
    
    # Ejecutar análisis
    agente.run_cycle(SYMBOLS)
