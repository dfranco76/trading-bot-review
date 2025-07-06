# src/main_bot.py - Versión con menú mejorado

from config import SYMBOLS
from sistema_multiagente import SistemaMultiAgente
from risk_manager import RiskManager
import time
from datetime import datetime
import schedule
import sys
import os

# Añadir paths para poder importar de utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

class TradingBotProfesional:
    def __init__(self):
        print("\n" + "="*60)
        print("🚀 TRADING BOT PROFESIONAL v5.0")
        print("="*60)
        
        # Inicializar componentes
        print("\n⚙️ Inicializando componentes...")
        self.risk_manager = RiskManager()
        self.sistema_multiagente = SistemaMultiAgente()
        
        print("\n✅ Bot profesional listo para operar")
        print("📊 5 agentes + Risk Manager activados")
        
    def verificar_condiciones_trading(self):
        """Verifica si es seguro operar"""
        # Verificar horario
        horario_ok, msg = self.risk_manager.verificar_horario_trading()
        if not horario_ok:
            print(f"\n⏰ {msg}")
            return False
        
        # Verificar límites de riesgo
        metricas = self.risk_manager.get_metricas_dia()
        if metricas['pnl_porcentaje'] <= -5:
            print(f"\n🛑 Límite de pérdida diaria alcanzado: {metricas['pnl_porcentaje']:.1f}%")
            return False
        
        return True
    
    def ejecutar_ciclo_trading(self):
        """Ejecuta un ciclo completo de análisis y trading"""
        print(f"\n{'='*60}")
        print(f"🔄 CICLO DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 1. Generar reporte de riesgo
        self.risk_manager.generar_reporte_riesgo_completo()
        
        # 2. Verificar condiciones
        if not self.verificar_condiciones_trading():
            return
        
        # 3. Ejecutar análisis multi-agente
        print("\n🤖 INICIANDO ANÁLISIS MULTI-AGENTE...")
        self.sistema_multiagente.execute_analysis(SYMBOLS)
        
        print(f"\n✅ Ciclo completado exitosamente")
    
    def generar_resumen_diario(self):
        """Genera resumen al final del día"""
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DIARIO - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Obtener métricas finales
        metricas = self.risk_manager.get_metricas_dia()
        
        print(f"\n📈 RESULTADOS DEL DÍA:")
        print(f"  • Total trades: {metricas['trades_totales']}")
        print(f"  • Ganados: {metricas['trades_ganadores']} | Perdidos: {metricas['trades_perdedores']}")
        print(f"  • Win Rate: {metricas['win_rate']*100:.0f}%")
        print(f"  • P&L: ${metricas['pnl_total']:.2f} ({metricas['pnl_porcentaje']:+.1f}%)")
        
        # Generar reporte de riesgo final
        self.risk_manager.generar_reporte_riesgo_completo()
    
    def modo_prueba(self):
        """Ejecuta una sola vez para probar"""
        print("\n🧪 MODO PRUEBA - Ejecutando un ciclo...")
        self.ejecutar_ciclo_trading()
        print("\n✅ Prueba completada")
    
    def modo_continuo(self):
        """Ejecuta continuamente cada 5 minutos"""
        print("\n🏃 MODO CONTINUO - Trading automático activado")
        print("⏹️ Presiona Ctrl+C para detener")
        
        # Programar tareas
        schedule.every(5).minutes.do(self.ejecutar_ciclo_trading)
        schedule.every().day.at("22:00").do(self.generar_resumen_diario)
        
        # Ejecutar primer ciclo inmediatamente
        self.ejecutar_ciclo_trading()
        
        # Loop principal
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n⏹️ Sistema detenido por el usuario")
                self.generar_resumen_diario()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("⏳ Reintentando en 60 segundos...")
                time.sleep(60)
    
    def modo_analisis_simple(self):
        """Solo analiza sin ejecutar trades"""
        print("\n🔍 MODO ANÁLISIS - Solo análisis sin trades")
        
        # Temporalmente deshabilitar ejecución de trades
        original_execute = self.sistema_multiagente.ejecutar_trade_profesional
        self.sistema_multiagente.ejecutar_trade_profesional = lambda x: print(f"  📝 [SIMULADO] {x['symbol']} {x['decision']}")
        
        self.ejecutar_ciclo_trading()
        
        # Restaurar
        self.sistema_multiagente.ejecutar_trade_profesional = original_execute
    
    def ejecutar_health_check(self):
        """Ejecuta el health check del sistema"""
        try:
            from health_check import main as health_check_main
            health_check_main()
        except ImportError:
            print("\n❌ No se encontró health_check.py en utils/")
            print("💡 Asegúrate de que el archivo existe en la carpeta utils")
    
    def ejecutar_verify_system(self):
        """Ejecuta la verificación completa del sistema"""
        try:
            from verify_system import main as verify_system_main
            verify_system_main()
        except ImportError:
            print("\n❌ No se encontró verify_system.py en utils/")
            print("💡 Asegúrate de que el archivo existe en la carpeta utils")
    
    def ejecutar_practice_mode(self):
        """Ejecuta el modo práctica sin restricciones"""
        try:
            # Importar y ejecutar directamente
            sys.path.insert(0, os.path.dirname(__file__))
            from practice_mode import main as practice_main
            practice_main()
        except ImportError:
            print("\n❌ No se encontró practice_mode.py")
            print("💡 Asegúrate de que el archivo existe en src/")


def mostrar_menu():
    """Muestra menú de opciones mejorado"""
    print("\n" + "="*60)
    print("🤖 TRADING BOT PROFESIONAL - MENÚ PRINCIPAL")
    print("="*60)
    
    print("\n📊 MODOS DE TRADING:")
    print("1. 🧪 Modo Prueba (1 ciclo con trades reales)")
    print("2. 🏃 Modo Continuo (automático cada 5 min)")
    print("3. 🔍 Modo Análisis (sin trades reales)")
    
    print("\n🔧 HERRAMIENTAS DE DIAGNÓSTICO:")
    print("4. 🏥 Health Check (verificar estado del sistema)")
    print("5. 🔍 Verify System (verificar todos los componentes)")
    print("6. 🎮 Practice Mode (análisis sin restricciones)")
    
    print("\n📈 REPORTES:")
    print("7. 📊 Ver Reporte de Riesgo Actual")
    print("8. 📋 Generar Resumen del Día")
    
    print("\n9. ❌ Salir")
    
    return input("\nSelecciona una opción (1-9): ")


if __name__ == "__main__":
    try:
        # Crear bot
        bot = TradingBotProfesional()
        
        while True:
            opcion = mostrar_menu()
            
            if opcion == "1":
                bot.modo_prueba()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "2":
                bot.modo_continuo()
                
            elif opcion == "3":
                bot.modo_analisis_simple()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "4":
                bot.ejecutar_health_check()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "5":
                bot.ejecutar_verify_system()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "6":
                bot.ejecutar_practice_mode()
                # practice_mode tiene su propio menú, no necesita esperar
                
            elif opcion == "7":
                bot.risk_manager.generar_reporte_riesgo_completo()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "8":
                bot.generar_resumen_diario()
                input("\nPresiona Enter para volver al menú...")
                
            elif opcion == "9":
                print("\n👋 ¡Hasta luego!")
                break
                
            else:
                print("\n❌ Opción no válida")
                
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        print("Por favor, verifica que todos los archivos estén correctamente configurados")