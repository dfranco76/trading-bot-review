# src/agente_pattern_recognition.py
from utils.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd

# Configuración Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class AgentePatternRecognition:
    def __init__(self):
        self.nombre = "Agente Pattern Recognition"
        self.capital = self.get_capital()
        print(f"✅ {self.nombre} iniciado con ${self.capital}")
    
    def get_capital(self):
        """Obtiene capital actual de la BD"""
        response = supabase.table('bot_status').select("capital").execute()
        return float(response.data[0]['capital'])
    
    def detectar_soporte_resistencia(self, data):
        """Detecta niveles de soporte y resistencia"""
        # Usar ventanas de diferentes períodos
        highs_5d = data['High'].rolling(window=5).max()
        lows_5d = data['Low'].rolling(window=5).min()
        highs_20d = data['High'].rolling(window=20).max()
        lows_20d = data['Low'].rolling(window=20).min()
        
        # Resistencias: máximos locales
        resistencia_corto = highs_5d.nlargest(3).mean()
        resistencia_largo = highs_20d.iloc[-1]
        
        # Soportes: mínimos locales
        soporte_corto = lows_5d.nsmallest(3).mean()
        soporte_largo = lows_20d.iloc[-1]
        
        return {
            'soporte_corto': soporte_corto,
            'soporte_largo': soporte_largo,
            'resistencia_corto': resistencia_corto,
            'resistencia_largo': resistencia_largo
        }
    
    def detectar_triangulo(self, data):
        """Detecta patrones de triángulo (convergencia de precios)"""
        if len(data) < 15:
            return None
            
        # Obtener máximos y mínimos de los últimos 15 días
        highs = data['High'][-15:].values
        lows = data['Low'][-15:].values
        
        # Calcular pendientes usando regresión lineal
        x = np.arange(len(highs))
        
        # Pendiente de máximos
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        # Pendiente de mínimos
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Calcular R² para ver qué tan bien se ajustan las líneas
        high_r2 = np.corrcoef(x, highs)[0, 1] ** 2
        low_r2 = np.corrcoef(x, lows)[0, 1] ** 2
        
        # Solo considerar si las líneas se ajustan bien (R² > 0.5)
        if high_r2 > 0.5 and low_r2 > 0.5:
            # Triángulo simétrico: ambas líneas convergen
            if abs(high_slope) > 0.001 and abs(low_slope) > 0.001:
                if high_slope < 0 and low_slope > 0:
                    return {
                        'tipo': 'TRIANGULO_SIMETRICO',
                        'confianza': min(high_r2, low_r2)
                    }
            
            # Triángulo ascendente: resistencia plana, soporte sube
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                return {
                    'tipo': 'TRIANGULO_ASCENDENTE',
                    'confianza': low_r2
                }
            
            # Triángulo descendente: soporte plano, resistencia baja
            if abs(low_slope) < 0.001 and high_slope < -0.001:
                return {
                    'tipo': 'TRIANGULO_DESCENDENTE',
                    'confianza': high_r2
                }
        
        return None
    
    def detectar_doble_suelo_techo(self, data):
        """Detecta patrones de doble suelo o doble techo"""
        if len(data) < 30:
            return None
            
        # Usar últimos 30 días
        prices = data['Close'][-30:]
        highs = data['High'][-30:]
        lows = data['Low'][-30:]
        
        # Encontrar mínimos y máximos locales significativos
        minimos = []
        maximos = []
        
        for i in range(2, len(prices)-2):
            # Mínimo local: menor que los 2 días anteriores y posteriores
            if all(lows.iloc[i] < lows.iloc[j] for j in [i-2, i-1, i+1, i+2] if j != i):
                minimos.append((i, lows.iloc[i]))
            
            # Máximo local: mayor que los 2 días anteriores y posteriores
            if all(highs.iloc[i] > highs.iloc[j] for j in [i-2, i-1, i+1, i+2] if j != i):
                maximos.append((i, highs.iloc[i]))
        
        # Doble suelo
        if len(minimos) >= 2:
            # Tomar los dos mínimos más recientes
            min1_idx, min1_val = minimos[-2]
            min2_idx, min2_val = minimos[-1]
            
            # Verificar que estén separados por al menos 5 días
            if min2_idx - min1_idx >= 5:
                diff_pct = abs(min1_val - min2_val) / min1_val
                
                # Si los mínimos son similares (diferencia < 3%)
                if diff_pct < 0.03:
                    # Verificar que haya un pico entre ellos
                    pico_entre = max(highs.iloc[min1_idx:min2_idx])
                    if pico_entre > min1_val * 1.03:  # Al menos 3% más alto
                        return {
                            'tipo': 'DOBLE_SUELO',
                            'nivel': (min1_val + min2_val) / 2,
                            'confianza': 1 - diff_pct  # Más confianza si son más similares
                        }
        
        # Doble techo
        if len(maximos) >= 2:
            # Tomar los dos máximos más recientes
            max1_idx, max1_val = maximos[-2]
            max2_idx, max2_val = maximos[-1]
            
            # Verificar que estén separados por al menos 5 días
            if max2_idx - max1_idx >= 5:
                diff_pct = abs(max1_val - max2_val) / max1_val
                
                # Si los máximos son similares (diferencia < 3%)
                if diff_pct < 0.03:
                    # Verificar que haya un valle entre ellos
                    valle_entre = min(lows.iloc[max1_idx:max2_idx])
                    if valle_entre < max1_val * 0.97:  # Al menos 3% más bajo
                        return {
                            'tipo': 'DOBLE_TECHO',
                            'nivel': (max1_val + max2_val) / 2,
                            'confianza': 1 - diff_pct
                        }
        
        return None
    
    def detectar_bandera(self, data):
        """Detecta patrones de bandera (flag)"""
        if len(data) < 20:
            return None
            
        # Separar en dos períodos: impulso (10 días) y consolidación (10 días)
        impulso = data['Close'][-20:-10]
        consolidacion = data['Close'][-10:]
        
        # Calcular características del impulso
        cambio_impulso = (impulso.iloc[-1] - impulso.iloc[0]) / impulso.iloc[0]
        volatilidad_impulso = impulso.std() / impulso.mean()
        
        # Calcular características de la consolidación
        cambio_consolidacion = (consolidacion.iloc[-1] - consolidacion.iloc[0]) / consolidacion.iloc[0]
        volatilidad_consolidacion = consolidacion.std() / consolidacion.mean()
        
        # Calcular pendiente de la consolidación
        x = np.arange(len(consolidacion))
        pendiente_consolidacion, _ = np.polyfit(x, consolidacion.values, 1)
        pendiente_pct = pendiente_consolidacion / consolidacion.mean()
        
        # Bandera alcista
        if cambio_impulso > 0.05:  # Impulso alcista fuerte (>5%)
            if volatilidad_consolidacion < volatilidad_impulso * 0.5:  # Consolidación
                if -0.02 < pendiente_pct < 0.01:  # Pendiente ligeramente negativa o plana
                    return {
                        'tipo': 'BANDERA_ALCISTA',
                        'impulso': cambio_impulso,
                        'confianza': min(cambio_impulso * 10, 0.8)
                    }
        
        # Bandera bajista
        elif cambio_impulso < -0.05:  # Impulso bajista fuerte (<-5%)
            if volatilidad_consolidacion < volatilidad_impulso * 0.5:  # Consolidación
                if -0.01 < pendiente_pct < 0.02:  # Pendiente ligeramente positiva o plana
                    return {
                        'tipo': 'BANDERA_BAJISTA',
                        'impulso': cambio_impulso,
                        'confianza': min(abs(cambio_impulso) * 10, 0.8)
                    }
        
        return None
    
    def detectar_hombro_cabeza_hombro(self, data):
        """Detecta el patrón hombro-cabeza-hombro"""
        if len(data) < 35:
            return None
            
        highs = data['High'][-35:]
        lows = data['Low'][-35:]
        
        # Buscar 3 picos (hombro-cabeza-hombro)
        picos = []
        for i in range(3, len(highs)-3):
            if all(highs.iloc[i] > highs.iloc[j] for j in [i-3, i-2, i-1, i+1, i+2, i+3] if j != i):
                picos.append((i, highs.iloc[i]))
        
        if len(picos) >= 3:
            # Tomar los 3 picos más recientes
            pico1 = picos[-3]
            pico2 = picos[-2]  # Cabeza
            pico3 = picos[-1]
            
            # Verificar proporciones del patrón
            # La cabeza debe ser más alta que los hombros
            if pico2[1] > pico1[1] * 1.02 and pico2[1] > pico3[1] * 1.02:
                # Los hombros deben ser similares (diferencia < 3%)
                diff_hombros = abs(pico1[1] - pico3[1]) / pico1[1]
                if diff_hombros < 0.03:
                    # Calcular línea de cuello (neckline)
                    valle1 = min(lows.iloc[pico1[0]:pico2[0]])
                    valle2 = min(lows.iloc[pico2[0]:pico3[0]])
                    neckline = (valle1 + valle2) / 2
                    
                    return {
                        'tipo': 'HOMBRO_CABEZA_HOMBRO',
                        'neckline': neckline,
                        'objetivo': neckline - (pico2[1] - neckline),  # Proyección bajista
                        'confianza': 0.75 * (1 - diff_hombros)
                    }
        
        return None
    
    def analyze_symbol(self, symbol):
        """Analiza un símbolo buscando patrones técnicos"""
        try:
            # Descargar datos
            stock = yf.Ticker(symbol)
            data = stock.history(period="2mo")  # 2 meses para mejor detección
            
            if len(data) < 30:
                return {'action': 'HOLD', 'confidence': 0.5}
            
            precio_actual = data['Close'].iloc[-1]
            volumen_promedio = data['Volume'][-20:].mean()
            volumen_actual = data['Volume'].iloc[-1]
            volumen_ratio = volumen_actual / volumen_promedio if volumen_promedio > 0 else 1
            
            print(f"  🔍 {symbol}: Analizando patrones...")
            
            # Detectar todos los patrones
            niveles = self.detectar_soporte_resistencia(data)
            triangulo = self.detectar_triangulo(data)
            doble = self.detectar_doble_suelo_techo(data)
            bandera = self.detectar_bandera(data)
            hch = self.detectar_hombro_cabeza_hombro(data)
            
            # Evaluar cada patrón y tomar decisión
            accion = 'HOLD'
            confianza = 0.5
            razon = ""
            
            # 1. Hombro-Cabeza-Hombro (patrón bajista fuerte)
            if hch:
                if precio_actual < hch['neckline']:
                    accion = 'SELL'
                    confianza = hch['confianza']
                    razon = f"Hombro-Cabeza-Hombro confirmado, objetivo: ${hch['objetivo']:.2f}"
            
            # 2. Doble techo/suelo
            elif doble:
                if doble['tipo'] == 'DOBLE_TECHO':
                    if precio_actual < doble['nivel'] * 0.98:
                        accion = 'SELL'
                        confianza = doble['confianza'] * 0.8
                        razon = f"Doble techo en ${doble['nivel']:.2f}"
                elif doble['tipo'] == 'DOBLE_SUELO':
                    if precio_actual > doble['nivel'] * 1.02:
                        accion = 'BUY'
                        confianza = doble['confianza'] * 0.8
                        razon = f"Doble suelo en ${doble['nivel']:.2f}"
            
            # 3. Banderas
            elif bandera:
                if bandera['tipo'] == 'BANDERA_ALCISTA':
                    accion = 'BUY'
                    confianza = bandera['confianza']
                    razon = f"Bandera alcista tras impulso de {bandera['impulso']*100:.1f}%"
                elif bandera['tipo'] == 'BANDERA_BAJISTA':
                    accion = 'SELL'
                    confianza = bandera['confianza']
                    razon = f"Bandera bajista tras caída de {bandera['impulso']*100:.1f}%"
            
            # 4. Triángulos
            elif triangulo:
                precio_medio = (data['High'].iloc[-1] + data['Low'].iloc[-1]) / 2
                if triangulo['tipo'] == 'TRIANGULO_ASCENDENTE':
                    if precio_actual > precio_medio:
                        accion = 'BUY'
                        confianza = triangulo['confianza'] * 0.7
                        razon = "Triángulo ascendente próximo a ruptura"
                elif triangulo['tipo'] == 'TRIANGULO_DESCENDENTE':
                    if precio_actual < precio_medio:
                        accion = 'SELL'
                        confianza = triangulo['confianza'] * 0.7
                        razon = "Triángulo descendente próximo a ruptura"
                elif triangulo['tipo'] == 'TRIANGULO_SIMETRICO' and volumen_ratio > 1.3:
                    # En triángulos simétricos, seguir la dirección del breakout
                    cambio_hoy = (precio_actual - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                    if cambio_hoy > 0.01:
                        accion = 'BUY'
                        confianza = triangulo['confianza'] * 0.65
                        razon = "Ruptura alcista de triángulo simétrico"
                    elif cambio_hoy < -0.01:
                        accion = 'SELL'
                        confianza = triangulo['confianza'] * 0.65
                        razon = "Ruptura bajista de triángulo simétrico"
            
            # 5. Rupturas de soporte/resistencia
            else:
                # Ruptura de resistencia
                if precio_actual > niveles['resistencia_largo'] * 0.99 and volumen_ratio > 1.5:
                    accion = 'BUY'
                    confianza = min(volumen_ratio / 3, 0.75)
                    razon = f"Ruptura de resistencia ${niveles['resistencia_largo']:.2f} con volumen"
                
                # Ruptura de soporte
                elif precio_actual < niveles['soporte_largo'] * 1.01 and volumen_ratio > 1.5:
                    accion = 'SELL'
                    confianza = min(volumen_ratio / 3, 0.75)
                    razon = f"Ruptura de soporte ${niveles['soporte_largo']:.2f} con volumen"
                
                # Rebote en soporte
                elif precio_actual > niveles['soporte_corto'] * 0.99 and precio_actual < niveles['soporte_corto'] * 1.02:
                    cambio_hoy = (precio_actual - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                    if cambio_hoy > 0.005 and volumen_ratio > 1.2:
                        accion = 'BUY'
                        confianza = 0.6
                        razon = f"Rebote en soporte ${niveles['soporte_corto']:.2f}"
            
            if razon:
                print(f"  ✨ Patrón detectado: {razon}")
            
            return {
                'action': accion,
                'confidence': confianza,
                'price': precio_actual,
                'reason': razon if razon else "Sin patrones claros"
            }
            
        except Exception as e:
            print(f"  ❌ Error analizando {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.5}
    
    def execute_trade(self, symbol, signal):
        """Registra trade en la BD"""
        if signal['action'] in ['BUY', 'SELL']:
            # Calcular cantidad (15% del capital para patrones)
            cantidad = (self.capital * 0.15) / signal['price']
            
            # Guardar en BD
            trade = {
                'symbol': symbol,
                'action': signal['action'],
                'quantity': cantidad,
                'price': signal['price'],
                'agent_confidence': signal['confidence'],
                'pattern_detected': signal['reason']
            }
            
            supabase.table('trades').insert(trade).execute()
            
            emoji = "📈" if signal['action'] == 'BUY' else "📉"
            print(f"{emoji} {signal['action']} (Patrón): {cantidad:.2f} {symbol} @ ${signal['price']:.2f}")
            print(f"   Razón: {signal['reason']}")
            print(f"   Confianza: {signal['confidence']*100:.0f}%")
    
    def run_cycle(self, symbols):
        """Ejecuta un ciclo de análisis"""
        print(f"\n⏰ Ciclo Pattern Recognition - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        encontro_algo = False
        for symbol in symbols:
            print(f"\nAnalizando {symbol}...")
            signal = self.analyze_symbol(symbol)
            
            if signal and signal['action'] != 'HOLD':
                self.execute_trade(symbol, signal)
                encontro_algo = True
        
        if not encontro_algo:
            print("\n🔍 No encontré patrones técnicos claros")
        
        print("\n✅ Ciclo completado")


if __name__ == "__main__":
    print("🤖 PATTERN RECOGNITION AGENT")
    print("Detectando patrones técnicos avanzados")
    print("="*50)
    
    # Crear agente
    agente = AgentePatternRecognition()
    
    # Símbolos a analizar
    from utils.config import SYMBOLS
    
    # Ejecutar análisis
    agente.run_cycle(SYMBOLS)