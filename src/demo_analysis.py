# src/demo_analysis.py
# MODO DEMO - Para ver el análisis sin restricciones de horario

from sistema_multiagente import SistemaMultiAgente
from config import SYMBOLS
from datetime import datetime

class DemoAnalysis:
    def __init__(self):
        print("\n" + "="*60)
        print("🎮 MODO DEMO - Análisis sin restricciones")
        print("="*60)
        print("⚠️ NOTA: Este es solo para ver cómo funciona el análisis")
        print("⚠️ NO ejecutará trades reales")
        
        self.sistema = SistemaMultiAgente()
        
        # Reemplazar la función de ejecutar trades
        self.sistema.ejecutar_trade_profesional = self.simular_trade
        
    def simular_trade(self, decision):
        """Simula un trade sin ejecutarlo"""
        print(f"\n📝 [SIMULACIÓN DE TRADE]:")
        print(f"  • Símbolo: {decision['symbol']}")
        print(f"  • Acción: {decision['decision']}")
        print(f"  • Tipo consenso: {decision['tipo']}")
        print(f"  • Confianza: {decision['confidence']*100:.0f}%")
        print(f"  • Precio actual: ${decision['price']:.2f}")
        
        # Simular cálculo de cantidad
        capital_simulado = 200
        tamano_posicion = 0.15 * decision['confidence']
        cantidad = (capital_simulado * tamano_posicion) / decision['price']
        
        print(f"  • Cantidad simulada: {cantidad:.2f} acciones")
        print(f"  • Capital que usaría: ${cantidad * decision['price']:.2f}")
        
        # Mostrar razones de los agentes
        print(f"\n  📝 Razones de los agentes:")
        for voto in decision['votos']:
            if voto['action'] == decision['decision']:
                print(f"     • {voto['agent']}: {voto['reason']}")
        
        return True
    
    def run(self):
        """Ejecuta el análisis demo"""
        print(f"\n🔍 Analizando {len(SYMBOLS)} símbolos...")
        print(f"Hora actual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ejecutar análisis
        self.sistema.execute_analysis(SYMBOLS)
        
        print("\n" + "="*60)
        print("✅ Análisis demo completado")
        print("💡 Recuerda: En trading real, el bot solo opera L-V 15:00-22:00")
        

if __name__ == "__main__":
    demo = DemoAnalysis()
    demo.run()
    
    input("\nPresiona Enter para salir...")