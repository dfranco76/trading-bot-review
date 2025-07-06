# safe_trading.py - Wrapper de seguridad para los primeros días
import sys
import os
sys.path.insert(0, 'src')

from sistema_multiagente import SistemaMultiAgente
from risk_manager import RiskManager
from datetime import datetime
import time

class SafeTradingWrapper:
    def __init__(self, modo='demo'):
        """
        Modos:
        - 'demo': Sin trades reales
        - 'micro': Trades de 20€ máximo
        - 'normal': Trades normales con límites estrictos
        """
        self.modo = modo
        self.sistema = SistemaMultiAgente()
        self.risk_manager = RiskManager()
        
        # Límites según modo
        self.limites = {
            'demo': {
                'max_trades_dia': 999,
                'max_posicion': 0,
                'max_perdida_dia': 0
            },
            'micro': {
                'max_trades_dia': 2,
                'max_posicion': 20,  # 20€ máximo
                'max_perdida_dia': 5  # 5€ máximo
            },
            'normal': {
                'max_trades_dia': 3,
                'max_posicion': 50,  # 50€ máximo
                'max_perdida_dia': 10  # 10€ máximo
            }
        }
        
        # Override ejecutar_trade
        self.original_ejecutar = self.sistema.ejecutar_trade_profesional
        self.sistema.ejecutar_trade_profesional = self.ejecutar_trade_seguro
        
        self.trades_hoy = 0
        self.perdida_hoy = 0
        
        print(f"🛡️ SAFE TRADING WRAPPER - Modo: {modo.upper()}")
        print(f"  • Max trades/día: {self.limites[modo]['max_trades_dia']}")
        print(f"  • Max posición: ${self.limites[modo]['max_posicion']}")
        print(f"  • Max pérdida/día: ${self.limites[modo]['max_perdida_dia']}")
    
    def ejecutar_trade_seguro(self, decision):
        """Wrapper seguro para ejecutar trades"""
        # Verificar límites
        if self.trades_hoy >= self.limites[self.modo]['max_trades_dia']:
            print(f"🛑 Límite de trades alcanzado ({self.trades_hoy})")
            return False
        
        if self.perdida_hoy >= self.limites[self.modo]['max_perdida_dia']:
            print(f"🛑 Límite de pérdida alcanzado (${self.perdida_hoy:.2f})")
            return False
        
        # Calcular tamaño seguro
        valor_trade = decision['quantity'] * decision['price']
        max_permitido = self.limites[self.modo]['max_posicion']
        
        if valor_trade > max_permitido and max_permitido > 0:
            # Ajustar cantidad
            decision['quantity'] = max_permitido / decision['price']
            print(f"⚠️ Tamaño ajustado a ${max_permitido} máximo")
        
        # Log detallado
        print(f"\n{'='*50}")
        print(f"🔍 EVALUANDO TRADE:")
        print(f"  • Símbolo: {decision['symbol']}")
        print(f"  • Acción: {decision['decision']}")
        print(f"  • Precio: ${decision['price']:.2f}")
        print(f"  • Cantidad: {decision['quantity']:.2f}")
        print(f"  • Valor: ${decision['quantity'] * decision['price']:.2f}")
        print(f"  • Confianza: {decision['confidence']*100:.0f}%")
        
        if self.modo == 'demo':
            print(f"  ✅ [DEMO] Trade simulado")
            # Registrar en paper trading reporter
            try:
                from paper_trading_reporter import reporter
                reporter.registrar_trade(decision)
            except:
                pass
            return True
        
        # Confirmación manual para los primeros trades
        if self.trades_hoy < 3 and self.modo != 'demo':
            respuesta = input("\n¿Ejecutar este trade? (s/n): ")
            if respuesta.lower() != 's':
                print("  ❌ Trade cancelado por usuario")
                return False
        
        # Ejecutar trade real
        resultado = self.original_ejecutar(decision)
        
        if resultado:
            self.trades_hoy += 1
            print(f"  ✅ Trade #{self.trades_hoy} ejecutado")
        
        return resultado
    
    def run_safe_cycle(self, symbols):
        """Ejecuta un ciclo con todas las protecciones"""
        print(f"\n🛡️ CICLO SEGURO - {datetime.now().strftime('%H:%M:%S')}")
        print(f"Modo: {self.modo.upper()} | Trades hoy: {self.trades_hoy}")
        
        # Verificar horario
        hora = datetime.now().hour
        if hora < 15 or hora >= 22:
            print("⏸️ Fuera de horario de trading")
            return
        
        # Ejecutar análisis
        self.sistema.execute_analysis(symbols)
    
    def quick_health_check(self):
        """Verificación rápida del sistema"""
        print("\n🏥 VERIFICACIÓN RÁPIDA")
        print("="*40)
        
        # Check capital
        capital = self.risk_manager.get_capital_actual()
        print(f"✅ Capital: ${capital:.2f}")
        
        # Check exposición
        exposicion = self.risk_manager.calcular_exposicion_actual()
        print(f"✅ Exposición: {exposicion['exposicion_total_pct']*100:.1f}%")
        
        # Check métricas del día
        metricas = self.risk_manager.get_metricas_dia()
        print(f"✅ Trades hoy: {metricas['trades_totales']}")
        print(f"✅ P&L: ${metricas['pnl_total']:.2f} ({metricas['pnl_porcentaje']:.1f}%)")
        
        return True
    
    def modo_supervision(self, symbols, intervalo_minutos=15):
        """Modo con supervisión continua"""
        print(f"\n👀 MODO SUPERVISIÓN - Intervalo: {intervalo_minutos} min")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                # Quick health check
                self.quick_health_check()
                
                # Ejecutar ciclo
                self.run_safe_cycle(symbols)
                
                # Esperar
                print(f"\n⏳ Próximo ciclo en {intervalo_minutos} minutos...")
                time.sleep(intervalo_minutos * 60)
                
        except KeyboardInterrupt:
            print("\n⏹️ Supervisión detenida")
            self.generar_resumen()
    
    def generar_resumen(self):
        """Genera resumen de la sesión"""
        print(f"\n{'='*60}")
        print("📊 RESUMEN DE LA SESIÓN")
        print(f"{'='*60}")
        print(f"Modo: {self.modo}")
        print(f"Trades ejecutados: {self.trades_hoy}")
        print(f"Pérdida acumulada: ${self.perdida_hoy:.2f}")
        print(f"Hora fin: {datetime.now().strftime('%H:%M:%S')}")


# Funciones helper para los diferentes días

def dia1_validacion():
    """Lunes - Solo validación"""
    from config import SYMBOLS
    
    print("\n🟦 DÍA 1 - VALIDACIÓN")
    print("="*60)
    
    wrapper = SafeTradingWrapper(modo='demo')
    
    # Ejecutar cada 30 minutos manualmente
    for i in range(10):  # 5 horas de trading
        print(f"\n\n>>> CICLO {i+1}/10")
        wrapper.run_safe_cycle(SYMBOLS)
        
        if i < 9:
            input("\n⏸️ Presiona Enter para siguiente ciclo (o Ctrl+C para salir)...")

def dia2_paper_trading():
    """Paper trading intensivo"""
    from config import SYMBOLS
    
    print("\n🟨 PAPER TRADING INTENSIVO")
    print("="*60)
    
    wrapper = SafeTradingWrapper(modo='demo')
    
    # Mensaje sobre reportes
    print("\n📊 REPORTES AUTOMÁTICOS:")
    print("  • Los trades se guardan automáticamente")
    print("  • Para ver reporte: python paper_trading_reporter.py reporte")
    print("  • Reporte final: python paper_trading_reporter.py final")
    
    wrapper.modo_supervision(SYMBOLS, intervalo_minutos=15)

def dia3_micro_test():
    """Test con 20€"""
    from config import SYMBOLS
    
    print("\n🟧 MICRO TEST (20€ máx)")
    print("="*60)
    print("⚠️ DINERO REAL - Máximo 2 trades de 20€")
    
    confirmacion = input("\n¿Confirmas que quieres operar con dinero real? (si/no): ")
    if confirmacion.lower() != 'si':
        print("❌ Cancelado")
        return
    
    wrapper = SafeTradingWrapper(modo='micro')
    wrapper.modo_supervision(SYMBOLS[:6], intervalo_minutos=30)  # Solo 6 símbolos

def dia4_trading_real():
    """Trading real con 200€"""
    from config import SYMBOLS
    
    print("\n🟥 TRADING REAL (200€)")
    print("="*60)
    print("⚠️ DINERO REAL - Límites estrictos activos")
    
    confirmacion = input("\n¿Confirmas que quieres operar con 200€ reales? (si/no): ")
    if confirmacion.lower() != 'si':
        print("❌ Cancelado")
        return
    
    # Verificación rápida
    wrapper = SafeTradingWrapper(modo='normal')
    if wrapper.quick_health_check():
        print("\n✅ Sistema verificado")
        
        continuar = input("\n¿Continuar con trading real? (si/no): ")
        if continuar.lower() == 'si':
            wrapper.modo_supervision(SYMBOLS, intervalo_minutos=20)
    else:
        print("\n❌ Verificación falló")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "dia1":
            dia1_validacion()
        elif sys.argv[1] == "dia2":
            dia2_paper_trading()
        elif sys.argv[1] == "dia3":
            dia3_micro_test()
        elif sys.argv[1] == "dia4":
            dia4_trading_real()
        else:
            print("Uso: python safe_trading.py [dia1|dia2|dia3|dia4]")
    else:
        print("\n🛡️ SAFE TRADING WRAPPER")
        print("="*60)
        print("\nUso:")
        print("  python safe_trading.py dia1  # Validación")
        print("  python safe_trading.py dia2  # Paper trading")
        print("  python safe_trading.py dia3  # Micro test 20€")
        print("  python safe_trading.py dia4  # Trading real 200€")
        print("\nO ejecuta directamente:")
        print("  python src/practice_mode.py  # Modo práctica")
        print("  python monitor.py            # Monitor de trading")