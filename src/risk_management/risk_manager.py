# risk_manager_enhanced.py
from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import numpy as np
import yfinance as yf
import pandas as pd

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class EnhancedRiskManager:
    def __init__(self, capital_inicial: float = 200):
        print("🛡️ ENHANCED RISK MANAGER v3.0")
        print("="*50)
        
        self.capital_inicial = capital_inicial
        
        # Límites de riesgo dinámicos
        self.base_limits = {
            'max_perdida_diaria': 0.05,
            'max_perdida_trade': 0.02,
            'max_drawdown_permitido': 0.10,
            'max_trades_dia': 15,
            'max_trades_hora': 5,
            'max_exposicion': 0.80,
            'max_exposicion_simbolo': 0.25,
            'max_correlacion': 0.70
        }
        
        # Copiar límites base como límites actuales
        self.limits = self.base_limits.copy()
        
        # Factores de ajuste adaptativos
        self.adaptive_factors = {
            'volatility_adjustment': 1.0,
            'drawdown_adjustment': 1.0,
            'streak_adjustment': 1.0,
            'market_regime_adjustment': 1.0
        }
        
        # Tracking de performance
        self.performance_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.daily_sharpe = 0
        self.current_drawdown = 0
        
        # Cache de correlaciones
        self.correlation_cache = {}
        self.cache_expiry = {}
        
        # Régimen de mercado
        self.market_regime = self.detectar_regimen_mercado()
        
        print(f"✅ Risk Manager mejorado iniciado")
        print(f"📊 Régimen de mercado detectado: {self.market_regime['regime']}")
    
    def get_capital_actual(self) -> float:
        """Obtiene el capital actual con cache"""
        if hasattr(self, '_capital_cache') and \
           hasattr(self, '_capital_cache_time') and \
           (datetime.now() - self._capital_cache_time).seconds < 60:
            return self._capital_cache
            
        response = supabase.table('bot_status').select("capital").execute()
        self._capital_cache = float(response.data[0]['capital']) if response.data else self.capital_inicial
        self._capital_cache_time = datetime.now()
        return self._capital_cache
    
    def detectar_regimen_mercado(self) -> Dict:
        """Detecta el régimen actual del mercado usando múltiples indicadores"""
        try:
            # Usar SPY como proxy del mercado
            spy = yf.Ticker('SPY')
            data = spy.history(period='3mo')
            
            if len(data) < 60:
                return {'regime': 'unknown', 'confidence': 0, 'volatility': 0.02}
            
            # Indicadores de tendencia
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            
            current_price = data['Close'].iloc[-1]
            
            # Volatilidad realizada
            returns = data['Close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # RSI para detectar extremos
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
            
            # Estructura del mercado
            higher_highs = 0
            lower_lows = 0
            for i in range(-20, -1):
                if data['High'].iloc[i] > data['High'].iloc[i-1]:
                    higher_highs += 1
                if data['Low'].iloc[i] < data['Low'].iloc[i-1]:
                    lower_lows += 1
            
            # Determinar régimen con múltiples criterios
            bull_score = 0
            bear_score = 0
            
            # Criterios de tendencia
            if current_price > sma_20.iloc[-1]:
                bull_score += 1
            else:
                bear_score += 1
                
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                bull_score += 1
            else:
                bear_score += 1
                
            if ema_12.iloc[-1] > ema_26.iloc[-1]:
                bull_score += 1
            else:
                bear_score += 1
            
            # Estructura de precios
            if higher_highs > lower_lows:
                bull_score += 1
            else:
                bear_score += 1
            
            # RSI
            if rsi > 50:
                bull_score += 0.5
            else:
                bear_score += 0.5
            
            # Determinar régimen
            total_score = bull_score + bear_score
            bull_percentage = bull_score / total_score
            
            if bull_percentage > 0.65:
                regime = 'bull'
                confidence = bull_percentage
            elif bull_percentage < 0.35:
                regime = 'bear'
                confidence = 1 - bull_percentage
            else:
                regime = 'sideways'
                confidence = 1 - abs(0.5 - bull_percentage) * 2
            
            # Ajustes por volatilidad extrema
            if volatility > 0.30:  # Volatilidad muy alta
                regime = 'volatile'
                confidence *= 0.8
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'rsi': rsi,
                'trend_strength': abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1],
                'bull_score': bull_score,
                'bear_score': bear_score
            }
            
        except Exception as e:
            print(f"Error detectando régimen de mercado: {e}")
            return {'regime': 'unknown', 'confidence': 0, 'volatility': 0.02}
    
    def calcular_correlacion_real(self, symbol1: str, symbol2: str, periodo: int = 30) -> float:
        """Calcula correlación real con cache"""
        cache_key = f"{symbol1}_{symbol2}_{periodo}"
        
        # Verificar cache
        if cache_key in self.correlation_cache:
            if cache_key in self.cache_expiry and datetime.now() < self.cache_expiry[cache_key]:
                return self.correlation_cache[cache_key]
        
        try:
            stock1 = yf.Ticker(symbol1)
            stock2 = yf.Ticker(symbol2)
            
            data1 = stock1.history(period=f"{periodo}d")['Close']
            data2 = stock2.history(period=f"{periodo}d")['Close']
            
            if len(data1) > 20 and len(data2) > 20:
                returns1 = data1.pct_change().dropna()
                returns2 = data2.pct_change().dropna()
                
                # Alinear las fechas
                common_dates = returns1.index.intersection(returns2.index)
                if len(common_dates) > 10:
                    correlation = returns1[common_dates].corr(returns2[common_dates])
                    
                    # Guardar en cache por 1 hora
                    self.correlation_cache[cache_key] = correlation
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                    
                    return correlation
        except:
            pass
        
        # Fallback a correlación aproximada
        return self.calcular_correlacion_aproximada(symbol1, symbol2)
    
    def calcular_correlacion_aproximada(self, simbolo1: str, simbolo2: str) -> float:
        """Calcula correlación aproximada mejorada por sectores"""
        sectores = {
            'semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'MU', 'AVGO', 'TSM'],
            'software': ['MSFT', 'CRM', 'ADBE', 'NOW', 'ORCL'],
            'cloud': ['NET', 'DDOG', 'SNOW', 'MDB', 'OKTA'],
            'fintech': ['SOFI', 'SQ', 'PYPL', 'V', 'MA', 'AFRM', 'UPST'],
            'ev': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV'],
            'crypto': ['COIN', 'MARA', 'RIOT', 'MSTR', 'CLSK', 'HIVE'],
            'ecommerce': ['AMZN', 'SHOP', 'MELI', 'SE', 'ETSY'],
            'social': ['META', 'SNAP', 'PINS', 'TWTR'],
            'streaming': ['NFLX', 'DIS', 'ROKU', 'PARA'],
            'gaming': ['NVDA', 'AMD', 'TTWO', 'EA', 'ATVI']
        }
        
        # Encontrar sectores
        sectors1 = []
        sectors2 = []
        
        for sector, symbols in sectores.items():
            if simbolo1 in symbols:
                sectors1.append(sector)
            if simbolo2 in symbols:
                sectors2.append(sector)
        
        # Calcular correlación base
        if not sectors1 or not sectors2:
            return 0.30  # Correlación base para símbolos desconocidos
        
        # Mismo sector exacto
        common_sectors = set(sectors1).intersection(set(sectors2))
        if common_sectors:
            if 'semiconductors' in common_sectors or 'crypto' in common_sectors:
                return 0.85  # Sectores muy correlacionados
            else:
                return 0.75
        
        # Sectores relacionados
        related_pairs = [
            ({'semiconductors', 'software'}, 0.60),
            ({'cloud', 'software'}, 0.65),
            ({'fintech', 'crypto'}, 0.55),
            ({'ev', 'semiconductors'}, 0.45),
            ({'ecommerce', 'fintech'}, 0.40),
            ({'social', 'streaming'}, 0.50)
        ]
        
        for pair_set, correlation in related_pairs:
            if (any(s in pair_set for s in sectors1) and 
                any(s in pair_set for s in sectors2)):
                return correlation
        
        # Sectores no relacionados
        return 0.25
    
    def ajustar_limites_dinamicamente(self):
        """Ajusta límites basado en performance y condiciones del mercado"""
        # Obtener métricas recientes
        metricas = self.get_metricas_dia()
        
        # 1. Ajuste por drawdown
        if self.current_drawdown < -5:
            self.adaptive_factors['drawdown_adjustment'] = 0.7
        elif self.current_drawdown < -3:
            self.adaptive_factors['drawdown_adjustment'] = 0.85
        else:
            self.adaptive_factors['drawdown_adjustment'] = 1.0
        
        # 2. Ajuste por racha
        if self.consecutive_losses >= 3:
            self.adaptive_factors['streak_adjustment'] = 0.6
        elif self.consecutive_losses >= 2:
            self.adaptive_factors['streak_adjustment'] = 0.8
        elif self.consecutive_wins >= 3:
            self.adaptive_factors['streak_adjustment'] = 1.1
        else:
            self.adaptive_factors['streak_adjustment'] = 1.0
        
        # 3. Ajuste por volatilidad del mercado
        if self.market_regime['volatility'] > 0.25:
            self.adaptive_factors['volatility_adjustment'] = 0.6
        elif self.market_regime['volatility'] > 0.20:
            self.adaptive_factors['volatility_adjustment'] = 0.8
        elif self.market_regime['volatility'] < 0.10:
            self.adaptive_factors['volatility_adjustment'] = 1.2
        else:
            self.adaptive_factors['volatility_adjustment'] = 1.0
        
        # 4. Ajuste por régimen de mercado
        regime_adjustments = {
            'bull': 1.1,
            'bear': 0.9,
            'sideways': 1.0,
            'volatile': 0.7,
            'unknown': 0.8
        }
        self.adaptive_factors['market_regime_adjustment'] = regime_adjustments.get(
            self.market_regime['regime'], 0.8
        )
        
        # Aplicar ajustes
        total_adjustment = 1.0
        for factor in self.adaptive_factors.values():
            total_adjustment *= factor
        
        # Limitar ajuste total
        total_adjustment = max(0.4, min(1.2, total_adjustment))
        
        # Actualizar límites
        self.limits['max_position_size'] = self.base_limits['max_exposicion_simbolo'] * total_adjustment
        self.limits['max_trades_dia'] = int(self.base_limits['max_trades_dia'] * total_adjustment)
        self.limits['max_exposicion'] = self.base_limits['max_exposicion'] * min(total_adjustment, 1.0)
    
    def calcular_var_historico(self, portfolio_positions: List[Dict], confidence_level: float = 0.95) -> float:
        """Calcula Value at Risk histórico del portfolio"""
        if not portfolio_positions:
            return 0
        
        try:
            # Obtener returns históricos de cada posición
            portfolio_returns = []
            weights = []
            
            for position in portfolio_positions:
                symbol = position['symbol']
                weight = position['value'] / self.get_capital_actual()
                
                stock = yf.Ticker(symbol)
                data = stock.history(period='3mo')
                
                if len(data) > 20:
                    returns = data['Close'].pct_change().dropna()
                    portfolio_returns.append(returns)
                    weights.append(weight)
            
            if not portfolio_returns:
                return 0
            
            # Calcular returns del portfolio
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalizar
            
            # Alinear fechas
            common_dates = portfolio_returns[0].index
            for returns in portfolio_returns[1:]:
                common_dates = common_dates.intersection(returns.index)
            
            # Calcular returns ponderados
            portfolio_return_series = sum(
                weight * returns[common_dates] 
                for weight, returns in zip(weights, portfolio_returns)
            )
            
            # Calcular VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_return_series, var_percentile)
            
            # Convertir a valor monetario
            capital_actual = self.get_capital_actual()
            var_amount = abs(var_value * capital_actual)
            
            return var_amount
            
        except Exception as e:
            print(f"Error calculando VaR: {e}")
            return self.get_capital_actual() * 0.05  # Default 5%
    
    def evaluar_trade_avanzado(self, simbolo: str, tipo: str, cantidad: float, precio: float, 
                              confianza: float, consenso_tipo: str, votos: List[Dict]) -> Tuple[bool, str, Dict]:
        """Evaluación avanzada con múltiples criterios y machine learning insights"""
        
        # Ajustar límites dinámicamente primero
        self.ajustar_limites_dinamicamente()
        
        ajustes = {
            'cantidad_original': cantidad,
            'cantidad_ajustada': cantidad,
            'factor_aplicado': 1.0,
            'razones_ajuste': [],
            'risk_score': 0,
            'expected_return': 0,
            'risk_reward_ratio': 0
        }
        
        capital_actual = self.get_capital_actual()
        valor_trade = cantidad * precio
        
        # 1. Verificaciones básicas (horario, límites duros)
        horario_ok, msg_horario = self.verificar_horario_avanzado()
        if not horario_ok:
            return False, msg_horario, ajustes
        
        # 2. Análisis de correlación con portfolio existente
        exposicion = self.calcular_exposicion_actual()
        if exposicion['simbolos_activos']:
            max_correlacion = 0
            for simbolo_existente in exposicion['simbolos_activos']:
                if simbolo_existente != simbolo:
                    correlacion = self.calcular_correlacion_real(simbolo, simbolo_existente)
                    max_correlacion = max(max_correlacion, abs(correlacion))
            
            if max_correlacion > self.limits['max_correlacion']:
                factor_correlacion = 1 - (max_correlacion - self.limits['max_correlacion'])
                ajustes['cantidad_ajustada'] *= factor_correlacion
                ajustes['razones_ajuste'].append(f"Alta correlación ({max_correlacion:.2f})")
        
        # 3. Análisis de volatilidad específica del símbolo
        try:
            stock = yf.Ticker(simbolo)
            hist = stock.history(period='1mo')
            if len(hist) > 10:
                returns = hist['Close'].pct_change().dropna()
                symbol_volatility = returns.std() * np.sqrt(252)
                
                if symbol_volatility > self.market_regime['volatility'] * 1.5:
                    vol_factor = self.market_regime['volatility'] / symbol_volatility
                    ajustes['cantidad_ajustada'] *= vol_factor
                    ajustes['razones_ajuste'].append(f"Alta volatilidad del símbolo ({symbol_volatility:.1%})")
        except:
            pass
        
        # 4. Análisis de sentimiento de los agentes
        agent_confidence_std = np.std([v['confidence'] for v in votos])
        if agent_confidence_std > 0.2:  # Alta divergencia entre agentes
            ajustes['cantidad_ajustada'] *= 0.8
            ajustes['razones_ajuste'].append("Alta divergencia entre agentes")
        
        # 5. Value at Risk check
        # Simular el portfolio con el nuevo trade
        simulated_positions = exposicion['exposicion_por_simbolo'].copy()
        simulated_positions[simbolo] = simulated_positions.get(simbolo, 0) + valor_trade
        
        simulated_portfolio = [
            {'symbol': s, 'value': v} 
            for s, v in simulated_positions.items()
        ]
        
        var_amount = self.calcular_var_historico(simulated_portfolio)
        var_percentage = var_amount / capital_actual
        
        if var_percentage > 0.10:  # VaR mayor al 10%
            var_factor = 0.10 / var_percentage
            ajustes['cantidad_ajustada'] *= var_factor
            ajustes['razones_ajuste'].append(f"VaR excesivo ({var_percentage:.1%})")
        
        # 6. Expected return vs risk
        # Calcular retorno esperado basado en el tipo de consenso y confianza
        expected_return_map = {
            'Consenso fuerte': 0.025,
            'Señal de expertos': 0.020,
            'Consenso moderado': 0.015,
            'Mayoría': 0.012,
            'Patrón técnico confirmado': 0.018
        }
        
        base_expected_return = expected_return_map.get(consenso_tipo, 0.010)
        expected_return = base_expected_return * confianza
        
        # Risk-reward ratio
        stop_loss_pct = self.limits['max_perdida_trade']
        risk_reward_ratio = expected_return / stop_loss_pct
        
        ajustes['expected_return'] = expected_return
        ajustes['risk_reward_ratio'] = risk_reward_ratio
        
        if risk_reward_ratio < 1.5:  # Ratio riesgo/beneficio bajo
            ajustes['cantidad_ajustada'] *= 0.7
            ajustes['razones_ajuste'].append(f"Ratio R:R bajo ({risk_reward_ratio:.2f})")
        
        # 7. Análisis de liquidez
        try:
            info = stock.info
            avg_volume = info.get('averageVolume', 0)
            if avg_volume > 0:
                trade_volume = cantidad
                volume_percentage = trade_volume / (avg_volume / 1000)  # Porcentaje del volumen diario
                
                if volume_percentage > 0.01:  # Más del 1% del volumen diario
                    liquidity_factor = 0.01 / volume_percentage
                    ajustes['cantidad_ajustada'] *= liquidity_factor
                    ajustes['razones_ajuste'].append(f"Impacto en liquidez ({volume_percentage:.1%} del volumen)")
        except:
            pass
        
        # 8. Machine Learning Risk Score (simulado por ahora)
        ml_risk_factors = {
            'market_regime_risk': 1.0 if self.market_regime['regime'] in ['bear', 'volatile'] else 0.5,
            'correlation_risk': max_correlacion if 'max_correlacion' in locals() else 0,
            'volatility_risk': symbol_volatility / 0.20 if 'symbol_volatility' in locals() else 1.0,
            'consensus_risk': 1.0 - confianza,
            'timing_risk': 0.8 if datetime.now().hour in [15, 21] else 0.3  # Primera y última hora
        }
        
        risk_score = np.mean(list(ml_risk_factors.values()))
        ajustes['risk_score'] = risk_score
        
        if risk_score > 0.7:
            ajustes['cantidad_ajustada'] *= (1 - risk_score/2)
            ajustes['razones_ajuste'].append(f"ML Risk Score alto ({risk_score:.2f})")
        
        # 9. Aplicar todos los factores adaptativos
        total_adaptive_factor = 1.0
        for factor in self.adaptive_factors.values():
            total_adaptive_factor *= factor
        
        ajustes['cantidad_ajustada'] *= total_adaptive_factor
        
        # 10. Verificaciones finales
        # Tamaño mínimo
        valor_minimo = capital_actual * 0.01  # Mínimo 1% del capital
        if ajustes['cantidad_ajustada'] * precio < valor_minimo:
            return False, f"Posición muy pequeña después de ajustes: ${ajustes['cantidad_ajustada'] * precio:.2f}", ajustes
        
        # Tamaño máximo
        valor_maximo = capital_actual * self.limits['max_position_size']
        if ajustes['cantidad_ajustada'] * precio > valor_maximo:
            ajustes['cantidad_ajustada'] = valor_maximo / precio
            ajustes['razones_ajuste'].append("Limitado por tamaño máximo de posición")
        
        # Calcular factor final
        ajustes['factor_aplicado'] = ajustes['cantidad_ajustada'] / cantidad
        
        # Mensaje final
        if ajustes['factor_aplicado'] < 0.5:
            mensaje = f"Trade aprobado con reducción significativa ({ajustes['factor_aplicado']:.1%})"
        elif ajustes['razones_ajuste']:
            mensaje = f"Trade aprobado con ajustes: {', '.join(ajustes['razones_ajuste'][:2])}"
        else:
            mensaje = "Trade aprobado sin ajustes"
        
        return True, mensaje, ajustes
    
    def verificar_horario_avanzado(self) -> Tuple[bool, str]:
        """Verificación de horario con consideraciones adicionales"""
        hora_actual = datetime.now().hour
        minuto_actual = datetime.now().minute
        dia_semana = datetime.now().weekday()
        
        # No tradear fines de semana
        if dia_semana >= 5:
            return False, "Mercado cerrado - Fin de semana"
        
        # Días especiales (añadir feriados importantes)
        fecha_actual = datetime.now().date()
        feriados_usa = [
            # Añadir feriados conocidos aquí
        ]
        
        if fecha_actual in feriados_usa:
            return False, "Mercado cerrado - Feriado en USA"
        
        # Horario normal
        if hora_actual < 15 or hora_actual >= 22:
            return False, f"Fuera de horario (15:00-22:00 España)"
        
        # Evitar primera media hora (alta volatilidad)
        if hora_actual == 15 and minuto_actual < 30:
            return False, "Primera media hora - Alta volatilidad"
        
        # Evitar últimos 15 minutos
        if hora_actual == 21 and minuto_actual > 45:
            return False, "Últimos 15 minutos - Cierre de posiciones institucionales"
        
        # Verificar si es día de Fed o eventos importantes
        # (implementar verificación de calendario económico)
        
        return True, "Horario óptimo para trading"
    
    def generar_reporte_riesgo_avanzado(self):
        """Genera reporte de riesgo completo con métricas avanzadas"""
        print("\n" + "="*70)
        print("📊 REPORTE DE RIESGO AVANZADO")
        print("="*70)
        
        # Capital y P&L
        capital = self.get_capital_actual()
        cambio_capital = ((capital - self.capital_inicial) / self.capital_inicial) * 100
        
        print(f"\n💰 CAPITAL Y PERFORMANCE:")
        print(f"  • Capital inicial: ${self.capital_inicial:.2f}")
        print(f"  • Capital actual: ${capital:.2f}")
        print(f"  • Cambio: {cambio_capital:+.1f}%")
        print(f"  • Drawdown actual: {self.current_drawdown:.1f}%")
        
        # Régimen de mercado
        print(f"\n🌍 CONDICIONES DE MERCADO:")
        print(f"  • Régimen: {self.market_regime['regime'].upper()}")
        print(f"  • Confianza: {self.market_regime['confidence']*100:.0f}%")
        print(f"  • Volatilidad: {self.market_regime['volatility']*100:.1f}%")
        print(f"  • RSI mercado: {self.market_regime.get('rsi', 50):.0f}")
        
        # Factores adaptativos
        print(f"\n🔧 AJUSTES ADAPTATIVOS:")
        for factor_name, factor_value in self.adaptive_factors.items():
            status = "🟢" if factor_value >= 1.0 else "🟡" if factor_value >= 0.8 else "🔴"
            print(f"  {status} {factor_name}: {factor_value:.2f}x")
        
        # Exposición y riesgo
        exposicion = self.calcular_exposicion_actual()
        print(f"\n💼 EXPOSICIÓN Y RIESGO:")
        print(f"  • Exposición total: {exposicion['exposicion_total_pct']*100:.1f}%")
        print(f"  • Símbolos activos: {len(exposicion['simbolos_activos'])}")
        
        if exposicion['simbolos_activos']:
            # Calcular VaR del portfolio
            positions = [
                {'symbol': s, 'value': v} 
                for s, v in exposicion['exposicion_por_simbolo'].items()
            ]
            var_95 = self.calcular_var_historico(positions, 0.95)
            var_99 = self.calcular_var_historico(positions, 0.99)
            
            print(f"  • VaR (95%): ${var_95:.2f} ({var_95/capital*100:.1f}%)")
            print(f"  • VaR (99%): ${var_99:.2f} ({var_99/capital*100:.1f}%)")
        
        # Correlaciones del portfolio
        if len(exposicion['simbolos_activos']) > 1:
            print(f"\n🔗 CORRELACIONES DEL PORTFOLIO:")
            correlations = []
            symbols = list(exposicion['simbolos_activos'])
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr = self.calcular_correlacion_real(symbols[i], symbols[j])
                    correlations.append((symbols[i], symbols[j], corr))
            
            # Mostrar top 3 correlaciones más altas
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            for s1, s2, corr in correlations[:3]:
                emoji = "⚠️" if abs(corr) > 0.7 else "✅"
                print(f"    {emoji} {s1}-{s2}: {corr:.2f}")
        
        # Límites actuales
        print(f"\n📏 LÍMITES ACTUALES (AJUSTADOS):")
        print(f"  • Max posición: {self.limits['max_position_size']*100:.1f}%")
        print(f"  • Max trades/día: {self.limits['max_trades_dia']}")
        print(f"  • Max exposición: {self.limits['max_exposicion']*100:.0f}%")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        
        if self.market_regime['regime'] == 'volatile':
            print("  ⚠️ Mercado volátil - Reducir tamaños de posición")
        
        if self.current_drawdown < -5:
            print("  ⚠️ Drawdown significativo - Considerar pausar trading")
        
        if exposicion['exposicion_total_pct'] > 0.6:
            print("  ⚡ Exposición alta - Ser selectivo con nuevas posiciones")
        
        if self.consecutive_losses >= 2:
            print("  ⚡ Racha negativa - Revisar estrategia y reducir riesgo")
        
        print("\n" + "="*70)