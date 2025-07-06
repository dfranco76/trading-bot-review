# src/agente_volume_momentum.py
from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

# Configuración Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AgenteVolumeMomentum:
    def __init__(self):
        self.nombre = "Agente Volume & Momentum"
        self.capital = self.get_capital()
        print(f"✅ {self.nombre} iniciado con ${self.capital}")
    
    def get_capital(self):
        """Obtiene capital actual de la BD"""
        response = supabase.table('bot_status').select("capital").execute()
        return float(response.data[0]['capital'])
    
    def calculate_obv(self, data):
        """Calcula On Balance Volume"""
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=data.index)
    
    def calculate_vwap(self, data, period=20):
        """Calcula Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).rolling(period).sum() / data['Volume'].rolling(period).sum()
        return vwap
    
    def calculate_mfi(self, data, period=14):
        """Calcula Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi.iloc[-1] if len(mfi) > 0 else 50
    
    def detect_volume_patterns(self, data):
        """Detecta patrones específicos de volumen"""
        patterns = []
        
        # Volumen promedio de diferentes períodos
        vol_5d = data['Volume'][-5:].mean()
        vol_20d = data['Volume'][-20:].mean()
        vol_actual = data['Volume'].iloc[-1]
        
        # Precio
        precio_actual = data['Close'].iloc[-1]
        cambio_precio = (precio_actual - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        
        # 1. Climax de volumen
        if vol_actual > vol_20d * 3:
            if cambio_precio > 0.02:
                patterns.append(('CLIMAX_COMPRA', 0.8))
            elif cambio_precio < -0.02:
                patterns.append(('CLIMAX_VENTA', 0.8))
        
        # 2. Volumen creciente (acumulación)
        if vol_5d > vol_20d * 1.5 and all(data['Volume'][-5:] > vol_20d):
            patterns.append(('ACUMULACION', 0.7))
        
        # 3. Volumen decreciente (distribución)
        if vol_5d < vol_20d * 0.5:
            patterns.append(('DISTRIBUCION', 0.6))
        
        # 4. Volumen en rango estrecho
        price_range = (data['High'][-5:].max() - data['Low'][-5:].min()) / precio_actual
        if price_range < 0.02 and vol_actual > vol_20d * 1.5:
            patterns.append(('COMPRESION_VOLUMEN', 0.75))
        
        return patterns
    
    def analyze_symbol(self, symbol):
        """Análisis avanzado de volumen y momentum"""
        try:
            # Descargar datos
            stock = yf.Ticker(symbol)
            data = stock.history(period="2mo")
            
            if len(data) < 20:
                return {'action': 'HOLD', 'confidence': 0.5}
            
            precio_actual = data['Close'].iloc[-1]
            volumen_actual = data['Volume'].iloc[-1]
            volumen_promedio_20d = data['Volume'][-20:].mean()
            volumen_promedio_5d = data['Volume'][-5:].mean()
            
            # Indicadores
            obv = self.calculate_obv(data)
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if obv.iloc[-5] != 0 else 0
            
            vwap = self.calculate_vwap(data)
            vwap_actual = vwap.iloc[-1]
            vwap_position = (precio_actual - vwap_actual) / vwap_actual
            
            mfi = self.calculate_mfi(data)
            
            # Ratios de volumen
            volumen_ratio = volumen_actual / volumen_promedio_20d if volumen_promedio_20d > 0 else 1
            volumen_trend = volumen_promedio_5d / volumen_promedio_20d if volumen_promedio_20d > 0 else 1
            
            # Momentum de precio
            cambio_1d = (precio_actual - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            cambio_5d = (precio_actual - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            
            # Detectar patrones
            patterns = self.detect_volume_patterns(data)
            
            # Análisis de spread (liquidez)
            spread = (data['High'].iloc[-1] - data['Low'].iloc[-1]) / precio_actual
            close_position = (precio_actual - data['Low'].iloc[-1]) / (data['High'].iloc[-1] - data['Low'].iloc[-1]) if data['High'].iloc[-1] != data['Low'].iloc[-1] else 0.5
            
            print(f"  📊 {symbol}: Volumen {volumen_ratio:.1f}x, Trend {volumen_trend:.1f}x")
            print(f"  📈 OBV Slope: {obv_slope*100:+.1f}%, MFI: {mfi:.0f}")
            print(f"  💹 VWAP: ${vwap_actual:.2f} (precio {vwap_position*100:+.1f}% arriba)")
            if patterns:
                print(f"  🎯 Patrones: {', '.join([p[0] for p in patterns])}")
            
            # ESTRATEGIAS AVANZADAS
            
            # 1. Explosión de volumen con dirección clara
            if volumen_ratio > 3 and abs(cambio_1d) > 0.02:
                if cambio_1d > 0 and close_position > 0.7:
                    return {
                        'action': 'BUY',
                        'confidence': min(0.85, volumen_ratio / 5),
                        'price': precio_actual,
                        'reason': f'Explosión de volumen {volumen_ratio:.1f}x con cierre fuerte ({close_position*100:.0f}% del rango)'
                    }
                elif cambio_1d < 0 and close_position < 0.3:
                    return {
                        'action': 'SELL',
                        'confidence': min(0.85, volumen_ratio / 5),
                        'price': precio_actual,
                        'reason': f'Volumen de pánico {volumen_ratio:.1f}x con cierre débil'
                    }
            
            # 2. Patrón de acumulación institucional
            elif 'ACUMULACION' in [p[0] for p in patterns] and obv_slope > 0.05:
                if precio_actual > vwap_actual and mfi < 70:
                    return {
                        'action': 'BUY',
                        'confidence': 0.75,
                        'price': precio_actual,
                        'reason': f'Acumulación institucional: OBV +{obv_slope*100:.1f}%, volumen trend {volumen_trend:.1f}x'
                    }
            
            # 3. Ruptura de VWAP con volumen y MFI
            elif precio_actual > vwap_actual * 1.01 and volumen_ratio > 1.5 and mfi > 50:
                if cambio_1d > 0.005:
                    return {
                        'action': 'BUY',
                        'confidence': 0.70,
                        'price': precio_actual,
                        'reason': f'Ruptura VWAP con volumen {volumen_ratio:.1f}x y MFI {mfi:.0f}'
                    }
            
            # 4. Compresión de volumen antes de movimiento
            elif 'COMPRESION_VOLUMEN' in [p[0] for p in patterns]:
                # Esperar dirección
                if cambio_1d > 0.01 and volumen_ratio > 2:
                    return {
                        'action': 'BUY',
                        'confidence': 0.72,
                        'price': precio_actual,
                        'reason': 'Ruptura de compresión con volumen explosivo'
                    }
            
            # 5. Divergencia OBV bajista
            elif cambio_5d > 0.05 and obv_slope < -0.05 and volumen_trend < 0.7:
                return {
                    'action': 'SELL',
                    'confidence': 0.70,
                    'price': precio_actual,
                    'reason': f'Divergencia bajista: precio +{cambio_5d*100:.1f}% pero OBV {obv_slope*100:.1f}%'
                }
            
            # 6. Distribución con MFI alto
            elif mfi > 80 and volumen_ratio > 2 and cambio_1d < -0.01:
                return {
                    'action': 'SELL',
                    'confidence': 0.75,
                    'price': precio_actual,
                    'reason': f'Distribución activa: MFI {mfi:.0f} con volumen alto'
                }
            
            # 7. Smart money: volumen bajo en retroceso
            elif cambio_5d < -0.03 and volumen_trend < 0.6 and cambio_1d > 0.01:
                return {
                    'action': 'BUY',
                    'confidence': 0.65,
                    'price': precio_actual,
                    'reason': 'Retroceso con volumen bajo - smart money acumulando'
                }
            
            return {'action': 'HOLD', 'confidence': 0.5}
            
        except Exception as e:
            print(f"  ❌ Error analizando {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.5}
    
    def execute_trade(self, symbol, signal):
        """Registra trade en la BD"""
        if signal['action'] in ['BUY', 'SELL']:
            # Calcular cantidad (18% del capital para volumen)
            cantidad = (self.capital * 0.18) / signal['price']
            
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
            
            emoji = "📊" if signal['action'] == 'BUY' else "📉"
            print(f"{emoji} {signal['action']} (Volume): {cantidad:.2f} {symbol} @ ${signal['price']:.2f}")
            print(f"   Razón: {signal['reason']}")
            print(f"   Confianza: {signal['confidence']*100:.0f}%")
    
    def run_cycle(self, symbols):
        """Ejecuta un ciclo de análisis"""
        print(f"\n⏰ Ciclo Volume & Momentum - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        encontro_algo = False
        for symbol in symbols:
            print(f"\nAnalizando {symbol}...")
            signal = self.analyze_symbol(symbol)
            
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                encontro_algo = True
        
        if not encontro_algo:
            print("\n🔍 No encontré patrones de volumen significativos")
        
        print("\n✅ Ciclo completado")


if __name__ == "__main__":
    print("📊 VOLUME & MOMENTUM AGENT")
    print("Analizando flujo de dinero institucional y patrones de volumen")
    print("="*50)
    
    # Crear agente
    agente = AgenteVolumeMomentum()
    
    # Símbolos a analizar
    from config import SYMBOLS
    
    # Ejecutar análisis
    agente.run_cycle(SYMBOLS)