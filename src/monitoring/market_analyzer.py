# market_analyzer.py
"""
Sistema avanzado de análisis de mercado en tiempo real
Detecta condiciones anormales, eventos importantes y oportunidades
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    SQUEEZE = "squeeze"
    BREAKOUT = "breakout"

@dataclass
class MarketEvent:
    timestamp: datetime
    event_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_symbols: List[str]
    action_required: bool

class MarketAnalyzer:
    def __init__(self):
        self.market_indicators = {}
        self.sector_performance = {}
        self.unusual_activity = []
        self.market_events = []
        
        # Índices principales para análisis
        self.market_indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'DIA': 'Dow Jones',
            'VIX': 'Volatility Index'
        }
        
        # Sectores principales
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        # Umbrales para detección de anomalías
        self.anomaly_thresholds = {
            'price_spike': 0.03,        # 3% en 5 minutos
            'volume_spike': 3.0,        # 3x volumen promedio
            'spread_widening': 0.005,   # 0.5% spread
            'correlation_break': 0.3,   # Cambio de 0.3 en correlación
            'volatility_spike': 2.0     # 2x volatilidad normal
        }
        
        print("📊 Market Analyzer iniciado")
    
    def analyze_market_breadth(self) -> Dict:
        """Analiza la amplitud del mercado (advance/decline, new highs/lows)"""
        try:
            # Simular análisis de amplitud con datos disponibles
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='1d', interval='1m')
            
            if len(spy_data) < 60:
                return {'status': 'insufficient_data'}
            
            # Calcular indicadores de amplitud
            price_changes = spy_data['Close'].pct_change()
            advancing_periods = sum(price_changes > 0)
            declining_periods = sum(price_changes < 0)
            
            # Advance/Decline ratio
            ad_ratio = advancing_periods / declining_periods if declining_periods > 0 else float('inf')
            
            # Momentum de amplitud
            recent_strength = price_changes.tail(30).mean()
            earlier_strength = price_changes.head(30).mean()
            breadth_momentum = recent_strength - earlier_strength
            
            # Determinar salud del mercado
            if ad_ratio > 2.0 and breadth_momentum > 0:
                market_health = "very_bullish"
            elif ad_ratio > 1.5:
                market_health = "bullish"
            elif ad_ratio < 0.5:
                market_health = "bearish"
            elif ad_ratio < 0.67:
                market_health = "very_bearish"
            else:
                market_health = "neutral"
            
            return {
                'advance_decline_ratio': ad_ratio,
                'breadth_momentum': breadth_momentum,
                'market_health': market_health,
                'advancing_periods': advancing_periods,
                'declining_periods': declining_periods,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error analizando amplitud del mercado: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def detect_sector_rotation(self) -> Dict:
        """Detecta rotación entre sectores"""
        try:
            sector_momentum = {}
            sector_relative_strength = {}
            
            # Obtener SPY como benchmark
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='1mo')
            spy_return_1w = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-5] - 1) if len(spy_data) >= 5 else 0
            spy_return_1m = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1)
            
            for symbol, sector in self.sector_etfs.items():
                try:
                    etf = yf.Ticker(symbol)
                    data = etf.history(period='1mo')
                    
                    if len(data) >= 20:
                        # Returns de diferentes períodos
                        return_1d = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) if len(data) >= 2 else 0
                        return_1w = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) if len(data) >= 5 else 0
                        return_1m = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)
                        
                        # Momentum (1w vs 1m)
                        momentum_score = return_1w - return_1m/4  # Momentum reciente vs largo plazo
                        
                        # Fuerza relativa vs SPY
                        relative_strength_1w = return_1w - spy_return_1w
                        relative_strength_1m = return_1m - spy_return_1m
                        
                        sector_momentum[sector] = {
                            'return_1d': return_1d,
                            'return_1w': return_1w,
                            'return_1m': return_1m,
                            'momentum_score': momentum_score,
                            'relative_strength_1w': relative_strength_1w,
                            'relative_strength_1m': relative_strength_1m
                        }
                except:
                    continue
            
            # Identificar rotaciones
            if not sector_momentum:
                return {'status': 'no_data'}
            
            # Ordenar por momentum
            sorted_sectors = sorted(
                sector_momentum.items(), 
                key=lambda x: x[1]['momentum_score'], 
                reverse=True
            )
            
            # Detectar rotación
            rotation_signals = []
            
            # Sectores ganando momentum
            for sector, metrics in sorted_sectors[:3]:
                if metrics['momentum_score'] > 0.01 and metrics['relative_strength_1w'] > 0:
                    rotation_signals.append({
                        'sector': sector,
                        'signal': 'entering',
                        'strength': metrics['momentum_score']
                    })
            
            # Sectores perdiendo momentum
            for sector, metrics in sorted_sectors[-3:]:
                if metrics['momentum_score'] < -0.01 and metrics['relative_strength_1w'] < 0:
                    rotation_signals.append({
                        'sector': sector,
                        'signal': 'exiting',
                        'strength': metrics['momentum_score']
                    })
            
            return {
                'sector_performance': sector_momentum,
                'rotation_signals': rotation_signals,
                'leading_sectors': [s[0] for s in sorted_sectors[:3]],
                'lagging_sectors': [s[0] for s in sorted_sectors[-3:]],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error detectando rotación de sectores: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def analyze_intermarket_relationships(self) -> Dict:
        """Analiza relaciones entre diferentes mercados"""
        relationships = {}
        
        # Pares importantes a monitorear
        intermarket_pairs = [
            ('TLT', 'SPY', 'Bonds vs Stocks'),
            ('GLD', 'SPY', 'Gold vs Stocks'),
            ('DXY', 'SPY', 'Dollar vs Stocks'),
            ('USO', 'SPY', 'Oil vs Stocks'),
            ('VIX', 'SPY', 'Volatility vs Stocks')
        ]
        
        try:
            for asset1, asset2, relationship_name in intermarket_pairs:
                try:
                    # Obtener datos
                    if asset1 == 'DXY':  # Dollar index no está en yfinance
                        continue
                        
                    ticker1 = yf.Ticker(asset1)
                    ticker2 = yf.Ticker(asset2)
                    
                    data1 = ticker1.history(period='3mo')['Close']
                    data2 = ticker2.history(period='3mo')['Close']
                    
                    if len(data1) > 60 and len(data2) > 60:
                        # Calcular correlaciones rolling
                        returns1 = data1.pct_change().dropna()
                        returns2 = data2.pct_change().dropna()
                        
                        # Alinear datos
                        common_dates = returns1.index.intersection(returns2.index)
                        returns1 = returns1[common_dates]
                        returns2 = returns2[common_dates]
                        
                        # Correlación de 20 días
                        corr_20d = returns1.tail(20).corr(returns2.tail(20))
                        # Correlación de 60 días
                        corr_60d = returns1.tail(60).corr(returns2.tail(60))
                        
                        # Detectar cambios en correlación
                        correlation_change = corr_20d - corr_60d
                        
                        # Performance relativa
                        perf_1w_asset1 = (data1.iloc[-1] / data1.iloc[-5] - 1) if len(data1) >= 5 else 0
                        perf_1w_asset2 = (data2.iloc[-1] / data2.iloc[-5] - 1) if len(data2) >= 5 else 0
                        relative_performance = perf_1w_asset1 - perf_1w_asset2
                        
                        relationships[relationship_name] = {
                            'correlation_20d': corr_20d,
                            'correlation_60d': corr_60d,
                            'correlation_change': correlation_change,
                            'relative_performance_1w': relative_performance,
                            'signal': self._interpret_intermarket_signal(
                                asset1, asset2, corr_20d, correlation_change, relative_performance
                            )
                        }
                except:
                    continue
            
            # Análisis general
            risk_on_indicators = 0
            risk_off_indicators = 0
            
            for rel_name, rel_data in relationships.items():
                if rel_data['signal'] == 'risk_on':
                    risk_on_indicators += 1
                elif rel_data['signal'] == 'risk_off':
                    risk_off_indicators += 1
            
            if risk_on_indicators > risk_off_indicators + 1:
                market_sentiment = "risk_on"
            elif risk_off_indicators > risk_on_indicators + 1:
                market_sentiment = "risk_off"
            else:
                market_sentiment = "neutral"
            
            return {
                'relationships': relationships,
                'market_sentiment': market_sentiment,
                'risk_on_indicators': risk_on_indicators,
                'risk_off_indicators': risk_off_indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error en análisis intermarket: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _interpret_intermarket_signal(self, asset1: str, asset2: str, correlation: float, 
                                     corr_change: float, rel_performance: float) -> str:
        """Interpreta señales de relaciones intermarket"""
        # Lógica específica por tipo de relación
        if asset1 == 'TLT' and asset2 == 'SPY':  # Bonds vs Stocks
            if correlation < -0.3 and rel_performance > 0:
                return "risk_off"  # Flight to quality
            elif correlation > 0.3:
                return "risk_on"   # Both rising = liquidity driven
                
        elif asset1 == 'VIX' and asset2 == 'SPY':  # Volatility vs Stocks
            if correlation < -0.7:
                return "normal"    # Relación inversa normal
            elif correlation > -0.3:
                return "warning"   # Correlación rompiendo = peligro
                
        elif asset1 == 'GLD' and asset2 == 'SPY':  # Gold vs Stocks
            if rel_performance > 0.05:
                return "risk_off"  # Gold outperforming = fear
            elif rel_performance < -0.05:
                return "risk_on"   # Stocks outperforming = greed
        
        return "neutral"
    
    def detect_market_anomalies(self, symbols: List[str]) -> List[Dict]:
        """Detecta anomalías en tiempo real en múltiples símbolos"""
        anomalies = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Datos de diferentes timeframes
                data_1d = ticker.history(period='1d', interval='1m')
                data_5d = ticker.history(period='5d', interval='5m')
                
                if len(data_1d) < 60:
                    continue
                
                # 1. Detección de spike de precio
                recent_prices = data_1d['Close'].tail(5)
                price_change_5m = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
                
                if abs(price_change_5m) > self.anomaly_thresholds['price_spike']:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'price_spike',
                        'severity': 'high' if abs(price_change_5m) > 0.05 else 'medium',
                        'value': price_change_5m,
                        'description': f"Movimiento de {price_change_5m*100:.1f}% en 5 minutos",
                        'timestamp': datetime.now()
                    })
                
                # 2. Detección de spike de volumen
                recent_volume = data_1d['Volume'].tail(5).mean()
                avg_volume = data_5d['Volume'].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > self.anomaly_thresholds['volume_spike']:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'volume_spike',
                        'severity': 'high' if volume_ratio > 5 else 'medium',
                        'value': volume_ratio,
                        'description': f"Volumen {volume_ratio:.1f}x el promedio",
                        'timestamp': datetime.now()
                    })
                
                # 3. Detección de aumento de volatilidad
                returns_recent = data_1d['Close'].pct_change().tail(30)
                returns_historical = data_5d['Close'].pct_change()
                
                vol_recent = returns_recent.std()
                vol_historical = returns_historical.std()
                vol_ratio = vol_recent / vol_historical if vol_historical > 0 else 1
                
                if vol_ratio > self.anomaly_thresholds['volatility_spike']:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'volatility_spike',
                        'severity': 'medium',
                        'value': vol_ratio,
                        'description': f"Volatilidad {vol_ratio:.1f}x lo normal",
                        'timestamp': datetime.now()
                    })
                
                # 4. Detección de gaps
                if len(data_5d) >= 2:
                    for i in range(1, min(5, len(data_5d))):
                        gap = (data_5d['Open'].iloc[-i] - data_5d['Close'].iloc[-i-1]) / data_5d['Close'].iloc[-i-1]
                        
                        if abs(gap) > 0.02:  # Gap mayor al 2%
                            anomalies.append({
                                'symbol': symbol,
                                'type': 'price_gap',
                                'severity': 'high' if abs(gap) > 0.05 else 'medium',
                                'value': gap,
                                'description': f"Gap de {gap*100:.1f}%",
                                'timestamp': data_5d.index[-i]
                            })
                            break
                
            except Exception as e:
                print(f"Error analizando {symbol}: {e}")
                continue
        
        return anomalies
    
    def analyze_market_microstructure(self, symbol: str) -> Dict:
        """Analiza la microestructura del mercado para un símbolo"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Obtener datos intraday
            data_1d = ticker.history(period='1d', interval='1m')
            
            if len(data_1d) < 60:
                return {'status': 'insufficient_data'}
            
            # 1. Análisis de spread efectivo
            high_low_spread = (data_1d['High'] - data_1d['Low']) / data_1d['Close']
            avg_spread = high_low_spread.mean()
            current_spread = high_low_spread.iloc[-1]
            
            # 2. Análisis de profundidad (aproximado con volumen)
            volume_profile = []
            price_levels = np.linspace(data_1d['Low'].min(), data_1d['High'].max(), 20)
            
            for i in range(len(price_levels)-1):
                mask = (data_1d['Close'] >= price_levels[i]) & (data_1d['Close'] < price_levels[i+1])
                vol_at_level = data_1d.loc[mask, 'Volume'].sum()
                volume_profile.append({
                    'price': (price_levels[i] + price_levels[i+1]) / 2,
                    'volume': vol_at_level
                })
            
            # Encontrar niveles de alto volumen (soporte/resistencia)
            volume_profile.sort(key=lambda x: x['volume'], reverse=True)
            high_volume_levels = volume_profile[:3]
            
            # 3. Análisis de toxicidad del flujo
            # (Detectar si hay mucho volumen en movimientos adversos)
            price_changes = data_1d['Close'].pct_change()
            volume_weighted_moves = (price_changes * data_1d['Volume']).sum() / data_1d['Volume'].sum()
            
            # 4. Detección de algoritmos
            # Buscar patrones de volumen consistentes (posible algo trading)
            volume_autocorr = data_1d['Volume'].autocorr(lag=1)
            likely_algo_trading = volume_autocorr > 0.7
            
            return {
                'average_spread': avg_spread,
                'current_spread': current_spread,
                'spread_widening': current_spread > avg_spread * 1.5,
                'high_volume_levels': high_volume_levels,
                'volume_weighted_direction': volume_weighted_moves,
                'likely_algo_trading': likely_algo_trading,
                'liquidity_score': 1 / (avg_spread * 100),  # Inverso del spread
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error en análisis de microestructura: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_market_event_calendar(self) -> List[MarketEvent]:
        """Obtiene eventos importantes del mercado"""
        events = []
        
        # Eventos regulares conocidos
        now = datetime.now()
        
        # FOMC meetings (simplificado - en producción usar API de calendario económico)
        fomc_dates = [
            # Añadir fechas FOMC conocidas
        ]
        
        # Earnings importantes esta semana
        # En producción, usar API de earnings
        
        # Vencimientos de opciones
        if now.weekday() == 4:  # Viernes
            # Tercer viernes del mes = vencimiento mensual
            if 15 <= now.day <= 21:
                events.append(MarketEvent(
                    timestamp=now,
                    event_type='options_expiry',
                    severity='high',
                    description='Monthly options expiration',
                    affected_symbols=['SPY', 'QQQ'],
                    action_required=True
                ))
        
        return events
    
    def generate_market_report(self) -> Dict:
        """Genera un reporte completo del estado del mercado"""
        print("\n🔍 Generando análisis de mercado completo...")
        
        report = {
            'timestamp': datetime.now(),
            'market_breadth': self.analyze_market_breadth(),
            'sector_rotation': self.detect_sector_rotation(),
            'intermarket_analysis': self.analyze_intermarket_relationships(),
            'anomalies': self.detect_market_anomalies(['SPY', 'QQQ', 'VIX']),
            'events': self.get_market_event_calendar()
        }
        
        # Resumen ejecutivo
        breadth = report['market_breadth']
        sectors = report['sector_rotation']
        intermarket = report['intermarket_analysis']
        
        # Determinar condición general del mercado
        if breadth.get('market_health') == 'very_bullish' and \
           intermarket.get('market_sentiment') == 'risk_on':
            overall_condition = MarketCondition.TRENDING_UP
        elif breadth.get('market_health') == 'very_bearish' and \
             intermarket.get('market_sentiment') == 'risk_off':
            overall_condition = MarketCondition.TRENDING_DOWN
        elif len(report['anomalies']) > 5:
            overall_condition = MarketCondition.VOLATILE
        else:
            overall_condition = MarketCondition.NORMAL
        
        report['executive_summary'] = {
            'overall_condition': overall_condition.value,
            'key_risks': self._identify_key_risks(report),
            'key_opportunities': self._identify_opportunities(report),
            'recommended_stance': self._recommend_trading_stance(report)
        }
        
        return report
    
    def _identify_key_risks(self, report: Dict) -> List[str]:
        """Identifica riesgos principales del mercado"""
        risks = []
        
        # Riesgo por amplitud
        if report['market_breadth'].get('market_health') in ['bearish', 'very_bearish']:
            risks.append("Amplitud de mercado débil - posible corrección")
        
        # Riesgo por anomalías
        high_severity_anomalies = [a for a in report['anomalies'] if a.get('severity') == 'high']
        if len(high_severity_anomalies) > 2:
            risks.append(f"Múltiples anomalías detectadas ({len(high_severity_anomalies)})")
        
        # Riesgo por rotación sectorial
        if report['sector_rotation'].get('rotation_signals'):
            defensive_rotating_in = any(
                s['sector'] in ['Utilities', 'Consumer Staples'] and s['signal'] == 'entering'
                for s in report['sector_rotation']['rotation_signals']
            )
            if defensive_rotating_in:
                risks.append("Rotación hacia sectores defensivos")
        
        return risks
    
    def _identify_opportunities(self, report: Dict) -> List[str]:
        """Identifica oportunidades en el mercado"""
        opportunities = []
        
        # Oportunidades por sectores fuertes
        if report['sector_rotation'].get('leading_sectors'):
            leading = report['sector_rotation']['leading_sectors']
            opportunities.append(f"Sectores líderes: {', '.join(leading[:2])}")
        
        # Oportunidades por condiciones favorables
        if report['market_breadth'].get('market_health') in ['bullish', 'very_bullish']:
            opportunities.append("Momentum positivo del mercado")
        
        return opportunities
    
    def _recommend_trading_stance(self, report: Dict) -> str:
        """Recomienda postura de trading basada en análisis"""
        risks = len(report['executive_summary']['key_risks'])
        opportunities = len(report['executive_summary']['key_opportunities'])
        
        if risks > opportunities + 1:
            return "defensive"
        elif opportunities > risks + 1:
            return "aggressive"
        else:
            return "neutral"
    
    def monitor_realtime(self, symbols: List[str], callback=None):
        """Monitorea el mercado en tiempo real"""
        print("🔴 Iniciando monitoreo en tiempo real...")
        
        while True:
            try:
                # Detectar anomalías
                anomalies = self.detect_market_anomalies(symbols)
                
                # Filtrar solo anomalías nuevas
                new_anomalies = []
                for anomaly in anomalies:
                    if not any(
                        a['symbol'] == anomaly['symbol'] and 
                        a['type'] == anomaly['type'] and
                        (anomaly['timestamp'] - a['timestamp']).seconds < 300
                        for a in self.unusual_activity
                    ):
                        new_anomalies.append(anomaly)
                        self.unusual_activity.append(anomaly)
                
                # Callback para anomalías nuevas
                if new_anomalies and callback:
                    for anomaly in new_anomalies:
                        callback(anomaly)
                
                # Limpiar anomalías viejas (más de 1 hora)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.unusual_activity = [
                    a for a in self.unusual_activity 
                    if a['timestamp'] > cutoff_time
                ]
                
                # Esperar antes de próximo ciclo
                import time
                time.sleep(60)  # Revisar cada minuto
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitoreo detenido")
                break
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                import time
                time.sleep(60)