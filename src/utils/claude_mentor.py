# src/claude_mentor.py
from anthropic import Anthropic
import json
from datetime import datetime
from typing import Dict, List, Optional
import time

class ClaudeMentor:
    def __init__(self, api_key: str):
        """Inicializa el mentor Claude"""
        self.client = Anthropic(api_key=api_key)
        # Usar claude-3-sonnet-20240229 que es más económico
        self.model = "claude-3-sonnet-20240229"
        self.max_retries = 3
        self.retry_delay = 2
        print("🧠 Claude Mentor iniciado")
        
    def analizar_decision_compleja(self, context: Dict) -> Dict:
        """Consulta a Claude para decisiones importantes"""
        
        # Construir prompt
        prompt = self._construir_prompt_trading(context)
        
        for intento in range(self.max_retries):
            try:
                # Llamar a Claude
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=0.3,  # Baja temperatura para consistencia
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                # Extraer el texto de la respuesta
                respuesta_texto = response.content[0].text
                
                # Parsear respuesta
                return self._parsear_respuesta_claude(respuesta_texto)
                
            except Exception as e:
                print(f"❌ Error consultando a Claude (intento {intento + 1}/{self.max_retries}): {e}")
                if intento < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        
        # Si todos los intentos fallan
        return {
            'aprobado': True,  # Por defecto aprobar para no bloquear el sistema
            'razon': 'No se pudo consultar a Claude - aprobado por defecto',
            'confianza_ajustada': context.get('confianza_original', 0.5),
            'riesgos_principales': [],
            'sugerencias': []
        }
    
    def _construir_prompt_trading(self, context: Dict) -> str:
        """Construye un prompt específico para trading"""
        
        symbol = context['symbol']
        action = context['action']
        votos = context['votos']
        indicadores = context.get('indicadores', {})
        capital_en_riesgo = context['capital_en_riesgo']
        
        prompt = f"""Eres un trader senior con 20 años de experiencia. Tu equipo de 5 agentes especializados quiere ejecutar este trade:

PROPUESTA DE TRADE:
- Símbolo: {symbol}
- Acción: {action}
- Capital en riesgo: ${capital_en_riesgo:.2f} ({context['porcentaje_capital']:.1f}% del capital total)

VOTOS DE LOS AGENTES ESPECIALIZADOS:
"""
        
        # Añadir votos con formato claro
        for voto in votos:
            emoji = "✅" if voto['action'] == action else "❌"
            prompt += f"{emoji} {voto['agent']}: {voto['action']} (confianza: {voto['confidence']*100:.0f}%) - {voto['reason']}\n"
        
        # Añadir resumen de consenso
        votos_favor = sum(1 for v in votos if v['action'] == action)
        votos_contra = sum(1 for v in votos if v['action'] != action and v['action'] != 'HOLD')
        
        prompt += f"""
RESUMEN DE CONSENSO:
- Votos a favor: {votos_favor}/5
- Votos en contra: {votos_contra}/5
- Confianza promedio del consenso: {context.get('confianza_consenso', 0)*100:.0f}%

CONTEXTO DE MERCADO:
- Día: {datetime.now().strftime('%A')}
- Hora: {datetime.now().strftime('%H:%M')} (Hora España)
"""

        # Añadir indicadores si están disponibles
        if indicadores:
            prompt += "\nINDICADORES TÉCNICOS:\n"
            for key, value in indicadores.items():
                if value != 'N/A':
                    prompt += f"- {key}: {value}\n"

        prompt += """
PREGUNTA: Como trader senior, ¿apruebas este trade?

Responde SOLO con un JSON válido en este formato exacto:
{
    "aprobado": true/false,
    "razon": "explicación breve de tu decisión",
    "confianza_ajustada": 0.0-1.0,
    "riesgos_principales": ["riesgo1", "riesgo2"],
    "sugerencias": ["sugerencia1", "sugerencia2"]
}

Sé conservador pero razonable. Considera el consenso de los agentes pero aplica tu experiencia.
"""
        
        return prompt
    
    def _parsear_respuesta_claude(self, respuesta: str) -> Dict:
        """Parsea la respuesta de Claude de forma robusta"""
        try:
            # Intentar extraer JSON de la respuesta
            import re
            
            # Buscar el JSON en la respuesta
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', respuesta, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                # Limpiar el JSON
                json_str = json_str.replace('true', 'True').replace('false', 'False')
                
                # Intentar parsear
                try:
                    resultado = eval(json_str)
                    
                    # Validar estructura
                    return {
                        'aprobado': bool(resultado.get('aprobado', True)),
                        'razon': str(resultado.get('razon', 'Sin razón especificada')),
                        'confianza_ajustada': float(resultado.get('confianza_ajustada', 0.5)),
                        'riesgos_principales': list(resultado.get('riesgos_principales', [])),
                        'sugerencias': list(resultado.get('sugerencias', []))
                    }
                except:
                    # Si falla eval, intentar con json.loads
                    json_str = json_match.group()
                    return json.loads(json_str)
            
            # Si no encuentra JSON, analizar la respuesta textual
            respuesta_lower = respuesta.lower()
            
            # Buscar palabras clave para determinar aprobación
            aprobado = True
            if any(palabra in respuesta_lower for palabra in ['no aprob', 'rechaz', 'no recomend', 'evitar', 'no ejecut']):
                aprobado = False
            elif any(palabra in respuesta_lower for palabra in ['aprob', 'adelante', 'ejecut', 'recomend']):
                aprobado = True
            
            # Extraer la razón (primeras 100 caracteres significativos)
            razon = respuesta.strip()[:200] if respuesta.strip() else "Respuesta no estructurada de Claude"
            
            return {
                'aprobado': aprobado,
                'razon': razon,
                'confianza_ajustada': 0.6 if aprobado else 0.3,
                'riesgos_principales': [],
                'sugerencias': []
            }
                
        except Exception as e:
            print(f"Error parseando respuesta de Claude: {e}")
            print(f"Respuesta original: {respuesta[:200]}...")
            
            return {
                'aprobado': True,  # Por defecto aprobar
                'razon': 'Error parseando respuesta - aprobado por defecto',
                'confianza_ajustada': 0.5,
                'riesgos_principales': [],
                'sugerencias': []
            }
    
    def validar_condiciones_mercado(self, condiciones: Dict) -> Dict:
        """Valida condiciones generales del mercado"""
        
        prompt = f"""Como experto en mercados, evalúa brevemente estas condiciones:

CONDICIONES ACTUALES:
- Día de la semana: {condiciones.get('dia_semana', 'N/A')}
- Volatilidad del mercado: {condiciones.get('volatilidad', 'Normal')}
- Eventos económicos hoy: {condiciones.get('eventos', 'Ninguno')}
- Trades ejecutados hoy: {condiciones.get('trades_hoy', 0)}
- P&L del día: {condiciones.get('pnl_dia', 0):.1f}%

¿Es un buen momento para trading activo? Responde solo con JSON:
{
    "trading_recomendado": true/false,
    "nivel_riesgo": "bajo/medio/alto",
    "razon": "explicación breve",
    "ajuste_sizing": 0.5-1.5
}"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parsear_respuesta_claude(response.content[0].text)
            
        except Exception as e:
            print(f"Error consultando condiciones de mercado: {e}")
            return {
                'trading_recomendado': True,
                'nivel_riesgo': 'medio',
                'razon': 'No se pudo consultar',
                'ajuste_sizing': 1.0
            }
    
    def generar_reporte_diario(self, trades_del_dia: List[Dict]) -> str:
        """Genera un análisis del día"""
        
        if not trades_del_dia:
            return "No hubo trades hoy para analizar."
        
        # Calcular estadísticas
        total_pnl = sum(t.get('pnl', 0) for t in trades_del_dia if t.get('pnl'))
        trades_ganadores = sum(1 for t in trades_del_dia if t.get('pnl', 0) > 0)
        trades_perdedores = sum(1 for t in trades_del_dia if t.get('pnl', 0) < 0)
        
        prompt = f"""Analiza brevemente el performance de trading de hoy:

RESUMEN DEL DÍA:
- Total trades: {len(trades_del_dia)}
- Ganadores: {trades_ganadores}
- Perdedores: {trades_perdedores}
- P&L total: ${total_pnl:.2f}

DETALLE DE TRADES:
"""
        for trade in trades_del_dia[:10]:  # Máximo 10 trades
            prompt += f"- {trade['symbol']}: {trade['action']} @ ${trade['price']:.2f}"
            if 'pnl' in trade:
                prompt += f" (P&L: ${trade['pnl']:.2f})"
            prompt += "\n"
        
        prompt += """
Proporciona un análisis ejecutivo breve (máximo 5 puntos) con:
1. Evaluación del performance
2. Principal acierto/error del día
3. Sugerencia clave para mañana
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generando reporte: {e}"


# Integración mejorada con el sistema principal
class ClaudeIntegration:
    def __init__(self, api_key: str):
        self.mentor = ClaudeMentor(api_key)
        self.umbral_consulta = 50  # Consultar si trade > $50
        self.trades_consultados = []
        self.consultas_hoy = 0
        self.max_consultas_dia = 100  # Límite para controlar costos
        
    def debe_consultar_claude(self, decision: Dict, capital: float) -> bool:
        """Determina si consultar a Claude"""
        
        # Verificar límite diario
        if self.consultas_hoy >= self.max_consultas_dia:
            print(f"⚠️ Límite diario de consultas a Claude alcanzado ({self.max_consultas_dia})")
            return False
        
        # Calcular valor del trade
        valor_trade = decision['quantity'] * decision['price']
        
        # Criterios para consultar (ordenados por prioridad)
        
        # 1. Trades grandes siempre se consultan
        if valor_trade > self.umbral_consulta * 2:  # >$100
            return True
        
        # 2. Consenso débil o controversial
        if decision.get('tipo', '').lower() in ['sin consenso', 'señal débil', 'mayoría simple']:
            return True
        
        # 3. Alta volatilidad o condiciones inusuales
        if decision.get('volatilidad_alta', False):
            return True
            
        # 4. Eventos importantes
        if decision.get('evento_importante', False):
            return True
            
        # 5. Señales contradictorias (ej: 3 BUY, 2 SELL)
        votos = decision.get('votos', [])
        if votos:
            acciones = [v['action'] for v in votos]
            if acciones.count('BUY') >= 2 and acciones.count('SELL') >= 2:
                return True
        
        # 6. Trades medianos con confianza media
        if valor_trade > self.umbral_consulta and decision.get('confidence', 0) < 0.7:
            return True
            
        return False
    
    def procesar_con_claude(self, decision: Dict, votos: List[Dict], indicadores: Dict) -> Dict:
        """Procesa una decisión con Claude"""
        
        # Preparar contexto completo
        context = {
            'symbol': decision['symbol'],
            'action': decision['decision'],
            'votos': votos,
            'indicadores': indicadores,
            'capital_en_riesgo': decision['quantity'] * decision['price'],
            'porcentaje_capital': (decision['quantity'] * decision['price'] / 200) * 100,
            'confianza_consenso': decision.get('confidence', 0.5),
            'confianza_original': decision.get('confidence', 0.5)
        }
        
        # Consultar a Claude
        print(f"  🤔 Consultando a Claude sobre {decision['symbol']}...")
        resultado = self.mentor.analizar_decision_compleja(context)
        
        # Incrementar contador
        self.consultas_hoy += 1
        
        # Registrar consulta
        self.trades_consultados.append({
            'timestamp': datetime.now(),
            'symbol': decision['symbol'],
            'decision_original': decision['decision'],
            'decision_claude': 'APROBADO' if resultado['aprobado'] else 'RECHAZADO',
            'razon': resultado['razon']
        })
        
        # Mostrar resultado
        if resultado['aprobado']:
            print(f"  ✅ Claude aprobó el trade")
            if resultado.get('sugerencias'):
                print(f"  💡 Sugerencias:")
                for sug in resultado['sugerencias'][:3]:  # Máximo 3 sugerencias
                    print(f"     • {sug}")
        else:
            print(f"  ❌ Claude rechazó el trade: {resultado['razon']}")
        
        # Ajustar decisión basado en Claude
        if not resultado['aprobado']:
            decision['action'] = 'HOLD'
            decision['decision'] = 'HOLD'
            decision['confidence'] *= 0.3  # Reducir confianza significativamente
            decision['claude_override'] = True
            decision['claude_reason'] = resultado['razon']
        else:
            # Claude aprobó - ajustar confianza si sugiere
            if resultado.get('confianza_ajustada'):
                decision['confidence'] = resultado['confianza_ajustada']
            decision['claude_approved'] = True
            decision['claude_suggestions'] = resultado.get('sugerencias', [])
            decision['claude_risks'] = resultado.get('riesgos_principales', [])
        
        return decision
    
    def obtener_estadisticas_claude(self) -> Dict:
        """Obtiene estadísticas de uso de Claude"""
        if not self.trades_consultados:
            return {
                'consultas_totales': 0,
                'aprobados': 0,
                'rechazados': 0,
                'tasa_aprobacion': 0
            }
        
        aprobados = sum(1 for t in self.trades_consultados if t['decision_claude'] == 'APROBADO')
        rechazados = len(self.trades_consultados) - aprobados
        
        return {
            'consultas_totales': len(self.trades_consultados),
            'consultas_hoy': self.consultas_hoy,
            'aprobados': aprobados,
            'rechazados': rechazados,
            'tasa_aprobacion': aprobados / len(self.trades_consultados)
        }


# Función de utilidad para test
def test_claude_mentor(api_key: str):
    """Función para probar la integración con Claude"""
    print("\n🧪 TEST DE CLAUDE MENTOR")
    print("="*50)
    
    try:
        # Crear mentor
        integration = ClaudeIntegration(api_key)
        
        # Crear decisión de prueba
        decision_test = {
            'symbol': 'NVDA',
            'decision': 'BUY',
            'action': 'BUY',
            'price': 850.00,
            'quantity': 0.1,
            'confidence': 0.65,
            'tipo': 'Consenso moderado'
        }
        
        votos_test = [
            {'agent': 'Momentum', 'action': 'BUY', 'confidence': 0.8, 'reason': 'Tendencia alcista fuerte'},
            {'agent': 'Mean Reversion', 'action': 'HOLD', 'confidence': 0.5, 'reason': 'Precio en rango normal'},
            {'agent': 'Pattern', 'action': 'BUY', 'confidence': 0.7, 'reason': 'Ruptura de resistencia'},
            {'agent': 'Volume', 'action': 'BUY', 'confidence': 0.6, 'reason': 'Volumen creciente'},
            {'agent': 'Sentiment', 'action': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment neutral'}
        ]
        
        indicadores_test = {
            'rsi': 65,
            'distancia_sma': 2.5,
            'volumen_ratio': 1.3
        }
        
        # Probar si debe consultar
        print(f"\n1. ¿Debe consultar Claude? {integration.debe_consultar_claude(decision_test, 200)}")
        
        # Procesar con Claude
        print("\n2. Procesando con Claude...")
        decision_procesada = integration.procesar_con_claude(
            decision_test.copy(),
            votos_test,
            indicadores_test
        )
        
        print(f"\n3. Resultado:")
        print(f"   - Decisión final: {decision_procesada.get('decision', 'HOLD')}")
        print(f"   - Confianza: {decision_procesada.get('confidence', 0)*100:.0f}%")
        print(f"   - Claude override: {decision_procesada.get('claude_override', False)}")
        
        # Mostrar estadísticas
        stats = integration.obtener_estadisticas_claude()
        print(f"\n4. Estadísticas:")
        print(f"   - Consultas: {stats['consultas_totales']}")
        print(f"   - Aprobados: {stats['aprobados']}")
        print(f"   - Rechazados: {stats['rechazados']}")
        
        print("\n✅ Test completado exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        print("Verifica que tu API key de Anthropic sea válida")


if __name__ == "__main__":
    # Para testing
    api_key = input("Ingresa tu API key de Anthropic (o 'skip' para saltar): ")
    if api_key and api_key != 'skip':
        test_claude_mentor(api_key)
    else:
        print("Test omitido")