# src/sistema_multiagente.py
from utils.config import SUPABASE_URL, SUPABASE_KEY, SYMBOLS
from supabase import create_client
import yfinance as yf
from datetime import datetime
from typing import Dict, List
import numpy as np
import json

# Importar TODOS los agentes especializados
from strategies.agente_momentum import AgenteMomentum
from strategies.agente_mean_reversion import AgenteMeanReversion
from strategies.agente_pattern_recognition import AgentePatternRecognition
from strategies.agente_volume_momentum import AgenteVolumeMomentum
from strategies.agente_sentiment import AgenteSentiment

# Intentar importar Claude (opcional)
try:
    from utils.claude_mentor import ClaudeIntegration
    from utils.config import ANTHROPIC_API_KEY, MAX_TRADE_SIZE
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ANTHROPIC_API_KEY = None
    MAX_TRADE_SIZE = 50

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SistemaMultiAgente:
    def __init__(self):
        print("🤖 SISTEMA MULTI-AGENTE PROFESIONAL v5.0")
        print("="*60)
        self.capital = self.get_capital()
        
        # Inicializar los 5 agentes especializados
        print("\n📊 Inicializando 5 agentes especializados...")
        self.agentes = [
            AgenteMomentum(),           # Tendencias y momentum
            AgenteMeanReversion(),      # Reversión a la media
            AgentePatternRecognition(), # Patrones técnicos
            AgenteVolumeMomentum(),     # Volumen y flujo de dinero
            AgenteSentiment()           # Sentiment y noticias
        ]
        
        # Inicializar Claude Mentor si está disponible
        self.claude_integration = None
        if CLAUDE_AVAILABLE and ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "tu_clave_aqui":
            try:
                print("\n🧠 Inicializando Claude Mentor...")
                self.claude_integration = ClaudeIntegration(ANTHROPIC_API_KEY)
                print("✅ Claude Mentor activado")
            except Exception as e:
                print(f"⚠️ Claude no disponible: {e}")
        else:
            print("\n⚠️ Claude Mentor no configurado")
        
        # === CONFIGURACIÓN AJUSTADA DEL SISTEMA DE CONSENSO ===
        # Umbrales más flexibles para encontrar más oportunidades
        self.umbral_consenso_fuerte = 0.55     # Reducido de 0.60 a 0.55
        self.umbral_consenso_moderado = 0.40   # Reducido de 0.45 a 0.40
        self.min_agentes_activos = 1           # Reducido de 2 a 1 (permite trades con 1 agente si confianza alta)
        self.min_confianza_single_agent = 0.80 # Nueva: confianza mínima para trade con 1 solo agente
        self.factor_kelly = 0.25                # Kelly Criterion conservador
        self.max_trades_simultaneos = 3        # Máximo 3 trades a la vez
        
        print(f"\n✅ Sistema iniciado con {len(self.agentes)} agentes especializados")
        print(f"💰 Capital disponible: ${self.capital:.2f}")
        print(f"⚙️ Configuración ajustada para más oportunidades")
        print(f"🤖 Recolección de datos ML activada")
        
    def get_capital(self):
        response = supabase.table('bot_status').select("capital").execute()
        return float(response.data[0]['capital']) if response.data else 200.0
    
    def obtener_voto_agente(self, agente, symbol):
        """Obtiene el voto de un agente de forma unificada"""
        try:
            # Todos los agentes tienen analyze_symbol()
            result = agente.analyze_symbol(symbol)
            
            if result and isinstance(result, dict):
                return {
                    'agent': agente.nombre,
                    'action': result.get('action', 'HOLD'),
                    'confidence': result.get('confidence', 0.5),
                    'reason': result.get('reason', 'Sin razón específica'),
                    'price': result.get('price', 0)
                }
                
        except Exception as e:
            print(f"  ❌ Error con {agente.nombre}: {e}")
        
        return {
            'agent': agente.nombre,
            'action': 'HOLD',
            'confidence': 0.5,
            'reason': 'Error o sin señal'
        }
    
    def extraer_features_para_ml(self, symbol, votos, decision):
        """Extrae features para futuros modelos de ML"""
        try:
            # Obtener datos del mercado
            stock = yf.Ticker(symbol)
            data = stock.history(period="1mo")
            
            if len(data) < 20:
                return None
            
            # Calcular indicadores técnicos básicos
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
            
            # Momentum
            mom_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] if len(data) >= 5 else 0
            mom_20d = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] if len(data) >= 20 else 0
            
            # Volumen
            volume_ratio = data['Volume'].iloc[-1] / data['Volume'][-20:].mean() if data['Volume'][-20:].mean() > 0 else 1
            
            # Volatilidad
            volatility = data['Close'][-20:].std() / data['Close'][-20:].mean() if len(data) >= 20 else 0.02
            
            # Distancia de medias móviles
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            distancia_sma20 = ((data['Close'].iloc[-1] - sma_20) / sma_20) * 100
            
            # Features de consenso
            votos_buy = sum(1 for v in votos if v['action'] == 'BUY')
            votos_sell = sum(1 for v in votos if v['action'] == 'SELL')
            
            # Confianzas promedio por acción
            conf_buy = np.mean([v['confidence'] for v in votos if v['action'] == 'BUY']) if votos_buy > 0 else 0
            conf_sell = np.mean([v['confidence'] for v in votos if v['action'] == 'SELL']) if votos_sell > 0 else 0
            
            # Preparar features
            features = {
                # Precio y volumen
                'price': float(data['Close'].iloc[-1]),
                'volume_ratio': float(volume_ratio),
                
                # Indicadores técnicos
                'rsi': float(rsi),
                'momentum_5d': float(mom_5d),
                'momentum_20d': float(mom_20d),
                'volatility_20d': float(volatility),
                'distance_sma20': float(distancia_sma20),
                
                # Estructura del mercado
                'high_low_ratio': float((data['High'].iloc[-1] - data['Low'].iloc[-1]) / data['Close'].iloc[-1]),
                'close_position': float((data['Close'].iloc[-1] - data['Low'].iloc[-1]) / 
                                      (data['High'].iloc[-1] - data['Low'].iloc[-1])) if data['High'].iloc[-1] != data['Low'].iloc[-1] else 0.5,
                
                # Features de consenso
                'votes_buy': int(votos_buy),
                'votes_sell': int(votos_sell),
                'votes_hold': int(5 - votos_buy - votos_sell),
                'avg_confidence_buy': float(conf_buy),
                'avg_confidence_sell': float(conf_sell),
                'consensus_confidence': float(decision.get('confidence', 0)),
                'unanimity': float(decision.get('unanimidad', 0)),
                
                # Votos individuales de cada agente
                'momentum_vote': next((v['action'] for v in votos if 'Momentum' in v['agent']), 'HOLD'),
                'momentum_conf': float(next((v['confidence'] for v in votos if 'Momentum' in v['agent']), 0.5)),
                'mean_rev_vote': next((v['action'] for v in votos if 'Mean Reversion' in v['agent']), 'HOLD'),
                'mean_rev_conf': float(next((v['confidence'] for v in votos if 'Mean Reversion' in v['agent']), 0.5)),
                'pattern_vote': next((v['action'] for v in votos if 'Pattern' in v['agent']), 'HOLD'),
                'pattern_conf': float(next((v['confidence'] for v in votos if 'Pattern' in v['agent']), 0.5)),
                'volume_vote': next((v['action'] for v in votos if 'Volume' in v['agent']), 'HOLD'),
                'volume_conf': float(next((v['confidence'] for v in votos if 'Volume' in v['agent']), 0.5)),
                'sentiment_vote': next((v['action'] for v in votos if 'Sentiment' in v['agent']), 'HOLD'),
                'sentiment_conf': float(next((v['confidence'] for v in votos if 'Sentiment' in v['agent']), 0.5)),
                
                # Metadata temporal
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'minutes_since_open': (datetime.now().hour - 15) * 60 + datetime.now().minute if datetime.now().hour >= 15 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"  ❌ Error extrayendo features para ML: {e}")
            return None
    
    def calcular_consenso_profesional(self, votos):
        """Calcula consenso usando múltiples métricas avanzadas - VERSIÓN MEJORADA"""
        # Filtrar votos válidos (umbral más bajo para incluir más señales)
        votos_validos = [v for v in votos if v['confidence'] > 0.35]  # Reducido de 0.4
        
        # === NUEVO: Permitir trades con 1 agente de alta confianza ===
        agentes_alta_confianza = [v for v in votos_validos if v['confidence'] >= self.min_confianza_single_agent and v['action'] != 'HOLD']
        
        if len(agentes_alta_confianza) >= 1:
            # Tomar el de mayor confianza
            mejor_agente = max(agentes_alta_confianza, key=lambda x: x['confidence'])
            return {
                'decision': mejor_agente['action'],
                'confidence': mejor_agente['confidence'] * 0.9,  # Pequeño descuento por ser solo 1 agente
                'tipo': f'Señal fuerte de {mejor_agente["agent"]}',
                'weights': {
                    'buy': 1.0 if mejor_agente['action'] == 'BUY' else 0,
                    'sell': 1.0 if mejor_agente['action'] == 'SELL' else 0,
                    'hold': 0
                },
                'counts': {
                    'buy': 1 if mejor_agente['action'] == 'BUY' else 0,
                    'sell': 1 if mejor_agente['action'] == 'SELL' else 0,
                    'hold': 0
                },
                'active_agents': 1,
                'unanimidad': 1.0,
                'avg_confidence': {
                    'buy': mejor_agente['confidence'] if mejor_agente['action'] == 'BUY' else 0,
                    'sell': mejor_agente['confidence'] if mejor_agente['action'] == 'SELL' else 0
                },
                'votos': votos  # Mantener todos los votos para referencia
            }
        
        # Si no hay agente único con alta confianza, continuar con el sistema normal
        if len(votos_validos) < self.min_agentes_activos:
            return {
                'decision': 'HOLD',
                'confidence': 0,
                'tipo': 'Datos insuficientes',
                'detalles': {},
                'votos': votos
            }
        
        # Agrupar por acción
        acciones = {'BUY': [], 'SELL': [], 'HOLD': []}
        for voto in votos_validos:
            acciones[voto['action']].append({
                'agent': voto['agent'],
                'confidence': voto['confidence'],
                'reason': voto['reason']
            })
        
        # Calcular métricas
        total_buy = sum(v['confidence'] for v in acciones['BUY'])
        total_sell = sum(v['confidence'] for v in acciones['SELL'])
        total_hold = sum(v['confidence'] for v in acciones['HOLD'])
        total = total_buy + total_sell + total_hold
        
        # Porcentajes ponderados
        pct_buy = total_buy / total if total > 0 else 0
        pct_sell = total_sell / total if total > 0 else 0
        pct_hold = total_hold / total if total > 0 else 0
        
        # Contar agentes activos (no HOLD o alta confianza)
        agentes_activos = sum(1 for v in votos if v['action'] != 'HOLD' or v['confidence'] > 0.6)
        
        # Calcular unanimidad y consistencia
        votos_buy = len(acciones['BUY'])
        votos_sell = len(acciones['SELL'])
        votos_hold = len(acciones['HOLD'])
        max_votos = max(votos_buy, votos_sell, votos_hold)
        unanimidad = max_votos / len(votos)
        
        # Calcular confianza promedio por acción
        avg_conf_buy = np.mean([v['confidence'] for v in acciones['BUY']]) if acciones['BUY'] else 0
        avg_conf_sell = np.mean([v['confidence'] for v in acciones['SELL']]) if acciones['SELL'] else 0
        
        # SISTEMA DE DECISIÓN MULTI-CRITERIO - AJUSTADO
        decision = 'HOLD'
        confianza_final = 0
        tipo_consenso = 'Sin consenso'
        
        # 1. Consenso fuerte (55%+ del peso) - AJUSTADO
        if pct_buy >= self.umbral_consenso_fuerte and agentes_activos >= 2:
            decision = 'BUY'
            confianza_final = pct_buy * (0.7 + 0.3 * unanimidad)
            tipo_consenso = 'Consenso fuerte alcista'
            
        elif pct_sell >= self.umbral_consenso_fuerte and agentes_activos >= 2:
            decision = 'SELL'
            confianza_final = pct_sell * (0.7 + 0.3 * unanimidad)
            tipo_consenso = 'Consenso fuerte bajista'
        
        # 2. Consenso moderado (40%+) - AJUSTADO
        elif pct_buy >= self.umbral_consenso_moderado and avg_conf_buy > 0.65:  # Reducido de 0.7
            decision = 'BUY'
            confianza_final = pct_buy * avg_conf_buy
            tipo_consenso = 'Consenso moderado alcista'
            
        elif pct_sell >= self.umbral_consenso_moderado and avg_conf_sell > 0.65:  # Reducido de 0.7
            decision = 'SELL'
            confianza_final = pct_sell * avg_conf_sell
            tipo_consenso = 'Consenso moderado bajista'
        
        # 3. Mayoría simple (2+ agentes) - AJUSTADO
        elif votos_buy >= 2 and unanimidad > 0.4:  # Reducido de 3 agentes y 0.6 unanimidad
            decision = 'BUY'
            confianza_final = avg_conf_buy * 0.85
            tipo_consenso = 'Mayoría alcista'
            
        elif votos_sell >= 2 and unanimidad > 0.4:  # Reducido de 3 agentes
            decision = 'SELL'
            confianza_final = avg_conf_sell * 0.85
            tipo_consenso = 'Mayoría bajista'
        
        # 4. Señal de expertos (2 agentes con alta confianza) - AJUSTADO
        elif votos_buy >= 2 and avg_conf_buy > 0.70:  # Reducido de 0.75
            decision = 'BUY'
            confianza_final = avg_conf_buy * 0.85
            tipo_consenso = 'Señal de expertos alcista'
            
        elif votos_sell >= 2 and avg_conf_sell > 0.70:  # Reducido de 0.75
            decision = 'SELL'
            confianza_final = avg_conf_sell * 0.85
            tipo_consenso = 'Señal de expertos bajista'
        
        # === NUEVO: Casos especiales para patrones fuertes ===
        # Si Pattern Recognition detecta algo con alta confianza
        pattern_votes = [v for v in votos if 'Pattern' in v['agent'] and v['confidence'] > 0.75]
        if pattern_votes and pattern_votes[0]['action'] != 'HOLD':
            if votos_buy >= 1 or votos_sell >= 1:  # Al menos otro agente apoya
                decision = pattern_votes[0]['action']
                confianza_final = pattern_votes[0]['confidence'] * 0.8
                tipo_consenso = 'Patrón técnico confirmado'
        
        # === NUEVO: Balancear cuando hay conflicto fuerte ===
        # Si hay señales opuestas fuertes (ej: 3 BUY vs 1 SELL con alta confianza)
        if votos_buy >= 3 and votos_sell == 1:
            # Verificar si el SELL es solo por sobrecompra técnica
            sell_agent = [v for v in votos if v['action'] == 'SELL'][0]
            if 'Mean Reversion' in sell_agent['agent'] and avg_conf_buy > 0.7:
                # Dar más peso a la mayoría si tienen alta confianza
                decision = 'BUY'
                confianza_final = avg_conf_buy * 0.75
                tipo_consenso = 'Mayoría alcista vs sobrecompra técnica'
        
        return {
            'decision': decision,
            'confidence': confianza_final,
            'tipo': tipo_consenso,
            'weights': {
                'buy': pct_buy,
                'sell': pct_sell,
                'hold': pct_hold
            },
            'counts': {
                'buy': votos_buy,
                'sell': votos_sell,
                'hold': votos_hold
            },
            'active_agents': agentes_activos,
            'unanimidad': unanimidad,
            'avg_confidence': {
                'buy': avg_conf_buy,
                'sell': avg_conf_sell
            },
            'votos': votos  # Incluir todos los votos
        }
    
    def calcular_tamano_posicion_avanzado(self, confianza, precio, volatilidad=None):
        """Calcula tamaño óptimo con Kelly Criterion y ajustes por volatilidad"""
        # Kelly básico
        probabilidad_exito = confianza
        probabilidad_fallo = 1 - confianza
        ratio_ganancia = 1.5  # Objetivo 1.5:1 reward/risk
        
        kelly_fraction = (probabilidad_exito * ratio_ganancia - probabilidad_fallo) / ratio_ganancia
        
        # Aplicar factor de seguridad
        kelly_conservador = kelly_fraction * self.factor_kelly
        
        # Ajustar por volatilidad si está disponible
        if volatilidad:
            if volatilidad > 0.03:  # Alta volatilidad
                kelly_conservador *= 0.7
            elif volatilidad < 0.015:  # Baja volatilidad
                kelly_conservador *= 1.2
        
        # Límites
        min_size = 0.05  # 5% mínimo
        max_size = 0.25  # 25% máximo
        
        return max(min_size, min(max_size, kelly_conservador))
    
    def mostrar_analisis_completo(self, symbol, votos, consenso):
        """Muestra análisis detallado de la votación"""
        print(f"\n{'='*60}")
        print(f"📊 ANÁLISIS COMPLETO: {symbol}")
        print(f"{'='*60}")
        
        # Votos individuales
        print("\n📋 VOTOS INDIVIDUALES:")
        for i, voto in enumerate(votos, 1):
            emoji = "🟢" if voto['action'] == 'BUY' else "🔴" if voto['action'] == 'SELL' else "⚪"
            confianza_bar = "█" * int(voto['confidence'] * 10) + "░" * (10 - int(voto['confidence'] * 10))
            
            print(f"\n  {i}. {voto['agent']}:")
            print(f"     {emoji} {voto['action']} [{confianza_bar}] {voto['confidence']*100:.0f}%")
            print(f"     💭 {voto['reason']}")
        
        # Resumen de consenso
        print(f"\n📈 ANÁLISIS DE CONSENSO:")
        print(f"  • Peso total: BUY={consenso['weights']['buy']*100:.0f}% | SELL={consenso['weights']['sell']*100:.0f}% | HOLD={consenso['weights']['hold']*100:.0f}%")
        print(f"  • Votos: BUY={consenso['counts']['buy']} | SELL={consenso['counts']['sell']} | HOLD={consenso['counts']['hold']}")
        print(f"  • Agentes activos: {consenso['active_agents']}/5")
        print(f"  • Unanimidad: {consenso['unanimidad']*100:.0f}%")
        print(f"  • Confianza promedio BUY: {consenso['avg_confidence']['buy']*100:.0f}%")
        print(f"  • Confianza promedio SELL: {consenso['avg_confidence']['sell']*100:.0f}%")
        
        # Decisión final
        emoji_decision = "🟢" if consenso['decision'] == 'BUY' else "🔴" if consenso['decision'] == 'SELL' else "⏸️"
        print(f"\n🎯 DECISIÓN FINAL:")
        print(f"  {emoji_decision} {consenso['decision']} - {consenso['tipo']}")
        print(f"  💪 Confianza del sistema: {consenso['confidence']*100:.0f}%")
        
        return consenso['decision'] != 'HOLD'
    
    def execute_analysis(self, symbols: List[str]):
        """Ejecuta análisis completo con los 5 agentes"""
        print(f"\n⏰ ANÁLISIS MULTI-AGENTE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        decisiones = []
        
        for symbol in symbols:
            print(f"\n\n🔍 ANALIZANDO {symbol}")
            print("-"*40)
            
            try:
                # Obtener precio actual para referencia
                stock = yf.Ticker(symbol)
                data = stock.history(period="1d")
                if len(data) == 0:
                    print(f"  ⚠️ No hay datos disponibles para {symbol}")
                    continue
                    
                precio_actual = data['Close'].iloc[-1]
                
                # Obtener voto de cada agente
                votos = []
                for agente in self.agentes:
                    voto = self.obtener_voto_agente(agente, symbol)
                    voto['price'] = precio_actual  # Asegurar precio
                    votos.append(voto)
                
                # Calcular consenso
                consenso = self.calcular_consenso_profesional(votos)
                
                # Mostrar análisis
                if self.mostrar_analisis_completo(symbol, votos, consenso):
                    consenso['symbol'] = symbol
                    consenso['price'] = precio_actual
                    consenso['quantity'] = (self.capital * 0.15) / precio_actual  # Cantidad base
                    decisiones.append(consenso)
                
            except Exception as e:
                print(f"  ❌ Error analizando {symbol}: {e}")
        
        # Ejecutar las mejores decisiones
        if decisiones:
            self.ejecutar_decisiones_optimizadas(decisiones)
        else:
            print(f"\n\n😴 No se encontraron oportunidades claras en este ciclo")
    
    def ejecutar_decisiones_optimizadas(self, decisiones):
        """Ejecuta las mejores decisiones con gestión de riesgo"""
        print(f"\n\n💼 EJECUTANDO DECISIONES")
        print("="*60)
        
        # Ordenar por confianza * tipo de consenso
        for d in decisiones:
            if 'fuerte' in d['tipo']:
                d['score'] = d['confidence'] * 1.2
            elif 'expertos' in d['tipo']:
                d['score'] = d['confidence'] * 1.1
            elif 'Señal fuerte de' in d['tipo']:  # Nuevo caso para agente único
                d['score'] = d['confidence'] * 1.05
            else:
                d['score'] = d['confidence']
        
        decisiones.sort(key=lambda x: x['score'], reverse=True)
        
        # Ejecutar máximo N trades
        trades_ejecutados = 0
        for decision in decisiones[:self.max_trades_simultaneos]:
            if self.ejecutar_trade_profesional(decision):
                trades_ejecutados += 1
        
        print(f"\n✅ {trades_ejecutados} trades ejecutados")
    
    def ejecutar_trade_profesional(self, decision):
        """Ejecuta un trade con toda la información y guarda features para ML"""
        try:
            # Calcular valor del trade
            valor_trade = decision['quantity'] * decision['price']
            
            # === NUEVO: Extraer features para ML ===
            features = self.extraer_features_para_ml(
                decision['symbol'], 
                decision.get('votos', []), 
                decision
            )
            
            # Consultar a Claude si está disponible y el trade es significativo
            if self.claude_integration and valor_trade > MAX_TRADE_SIZE:
                print(f"\n🧠 Consultando a Claude Mentor (trade > ${MAX_TRADE_SIZE})...")
                
                # Preparar contexto para Claude
                indicadores = features if features else {}
                
                # Procesar con Claude
                decision = self.claude_integration.procesar_con_claude(
                    decision, 
                    decision['votos'],
                    indicadores
                )
                
                # Si Claude rechazó el trade
                if decision.get('claude_override'):
                    print(f"❌ Claude rechazó el trade: {decision['claude_reason']}")
                    return False
                elif decision.get('claude_approved'):
                    print(f"✅ Claude aprobó el trade")
                    if decision.get('claude_suggestions'):
                        print("💡 Sugerencias de Claude:")
                        for sug in decision['claude_suggestions']:
                            print(f"   • {sug}")
            
            # Calcular tamaño óptimo
            tamano_posicion = self.calcular_tamano_posicion_avanzado(
                decision['confidence'],
                decision['price']
            )
            
            cantidad = (self.capital * tamano_posicion) / decision['price']
            
            # Preparar registro completo
            trade_record = {
                'symbol': decision['symbol'],
                'action': decision['decision'],
                'quantity': cantidad,
                'price': decision['price'],
                'agent_confidence': decision['confidence'],
                'consensus_type': decision['tipo'],
                'position_size': tamano_posicion,
                'buy_votes': decision['counts']['buy'],
                'sell_votes': decision['counts']['sell'],
                'unanimidad': decision['unanimidad'],
                'claude_consulted': decision.get('claude_approved', False),
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar razones principales
            razones = []
            for voto in decision['votos']:
                if voto['action'] == decision['decision']:
                    razones.append({
                        'agent': voto['agent'],
                        'reason': voto['reason'],
                        'confidence': voto['confidence']
                    })
            
            # Insertar trade en BD
            trade_response = supabase.table('trades').insert(trade_record).execute()
            
            # === NUEVO: Guardar features para ML ===
            if features and trade_response.data:
                trade_id = trade_response.data[0]['id']
                
                ml_features_record = {
                    'trade_id': trade_id,
                    'symbol': decision['symbol'],
                    'timestamp': datetime.now().isoformat(),
                    'features': json.dumps(features),  # Guardar como JSON
                    'ml_prediction': decision['decision'],  # Por ahora es la decisión del sistema de reglas
                    'ml_confidence': decision['confidence'],
                    'ml_model_version': 'rules_v1'  # Versión actual basada en reglas
                }
                
                # Intentar insertar features para ML
                try:
                    supabase.table('ml_features').insert(ml_features_record).execute()
                    print(f"  🤖 Features ML guardados para futuro entrenamiento")
                except:
                    # Si la tabla no existe, no es crítico
                    pass
            
            # Mostrar ejecución
            emoji = "🚀" if decision['decision'] == 'BUY' else "💰"
            print(f"\n{emoji} TRADE EJECUTADO:")
            print(f"  • Símbolo: {decision['symbol']}")
            print(f"  • Acción: {decision['decision']}")
            print(f"  • Cantidad: {cantidad:.2f} acciones")
            print(f"  • Precio: ${decision['price']:.2f}")
            print(f"  • Capital usado: ${cantidad * decision['price']:.2f} ({tamano_posicion*100:.0f}%)")
            print(f"  • Tipo consenso: {decision['tipo']}")
            print(f"  • Confianza: {decision['confidence']*100:.0f}%")
            if decision.get('claude_approved'):
                print(f"  • ✅ Validado por Claude")
            
            print(f"\n  📝 Razones principales:")
            for r in sorted(razones, key=lambda x: x['confidence'], reverse=True)[:3]:
                print(f"     • {r['agent']}: {r['reason']} ({r['confidence']*100:.0f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error ejecutando trade: {e}")
            return False


if __name__ == "__main__":
    sistema = SistemaMultiAgente()
    sistema.execute_analysis(SYMBOLS)
    
    print("\n\n✅ Análisis multi-agente completado")
    print("📊 Revisa Supabase para ver los trades ejecutados")
    print("🤖 Features ML guardados para futuro entrenamiento")