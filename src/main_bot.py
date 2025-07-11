
def safe_input(prompt="Presiona Enter para continuar..."):
    """Input que permite salir con q o 9"""
    response = input(prompt)
    if response.lower() in ['q', '9', 'quit', 'exit']:
        print("\n👋 ¡Hasta luego!")
        import sys
        sys.exit(0)
    return response

#!/usr/bin/env python3
"""
Trading Bot Professional - Main Entry Point
Handles the orchestration of multiple trading agents with risk management.
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import schedule
import logging
import importlib.util
from typing import Optional, Callable

# Configure project paths properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
UTILS_DIR = PROJECT_ROOT / 'utils'

# Add necessary paths to Python path
for path in [PROJECT_ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'not set')}")
except ImportError:
    logger.warning("python-dotenv not installed, using system environment variables")

# Import project modules
from src.utils.config import SYMBOLS
from src.strategies.sistema_multiagente import SistemaMultiAgente
from src.risk_management.risk_manager import EnhancedRiskManager


class ModuleLoader:
    """Handles dynamic module loading following best practices"""
    
    @staticmethod
    def load_module_from_path(module_name: str, file_path: Path) -> Optional[object]:
        """
        Dynamically load a Python module from a file path.
        
        Args:
            module_name: Name to assign to the module
            file_path: Path to the Python file
            
        Returns:
            Loaded module object or None if failed
        """
        try:
            if not file_path.exists():
                logger.error(f"Module file not found: {file_path}")
                return None
                
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec for: {file_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            
            # Set important module attributes
            module.__file__ = str(file_path)
            module.__name__ = module_name
            
            # Add project paths to the module's namespace
            module.__dict__['PROJECT_ROOT'] = str(PROJECT_ROOT)
            module.__dict__['SRC_DIR'] = str(SRC_DIR)
            
            # Temporarily add necessary paths
            original_path = sys.path.copy()
            paths_to_add = [str(UTILS_DIR), str(PROJECT_ROOT), str(SRC_DIR)]
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            try:
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"Successfully loaded module: {module_name} from {file_path}")
                return module
            finally:
                # Restore original path
                sys.path = original_path
            
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def execute_module_main(module_name: str, file_path: Path) -> bool:
        """
        Load a module and execute its main() function.
        
        Args:
            module_name: Name of the module
            file_path: Path to the module file
            
        Returns:
            True if executed successfully, False otherwise
        """
        module = ModuleLoader.load_module_from_path(module_name, file_path)
        
        if module is None:
            return False
            
        if not hasattr(module, 'main'):
            logger.error(f"Module {module_name} does not have a main() function")
            return False
            
        try:
            module.main()
            return True
        except Exception as e:
            logger.error(f"Error executing {module_name}.main(): {e}", exc_info=True)
            return False


class TradingBotProfesional:
    """Professional trading bot with multi-agent system and risk management"""
    
    def __init__(self):
        """Initialize the trading bot components"""
        self._print_header()
        
        try:
            logger.info("Initializing trading bot components...")
            self.risk_manager = EnhancedRiskManager()
            self.sistema_multiagente = SistemaMultiAgente()
            self.module_loader = ModuleLoader()
            
            print("\n✅ Bot profesional listo para operar")
            print("📊 5 agentes + Risk Manager activados")
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}", exc_info=True)
            raise
    
    def _print_header(self):
        """Print bot header"""
        print("\n" + "="*60)
        print("🚀 TRADING BOT PROFESIONAL v5.0")
        print("="*60)
        print("\n⚙️ Inicializando componentes...")
    
    def verificar_condiciones_trading(self) -> bool:
        """
        Verify if trading conditions are safe.
        
        Returns:
            True if safe to trade, False otherwise
        """
        try:
            # Check trading hours
            horario_ok, msg = self.risk_manager.verificar_horario_trading()
            if not horario_ok:
                print(f"\n⏰ {msg}")
                logger.warning(f"Trading hours check failed: {msg}")
                return False
            
            # Check risk limits
            metricas = self.risk_manager.get_metricas_dia()
            if metricas['pnl_porcentaje'] <= -5:
                msg = f"Daily loss limit reached: {metricas['pnl_porcentaje']:.1f}%"
                print(f"\n🛑 {msg}")
                logger.warning(msg)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {e}", exc_info=True)
            return False
    
    def ejecutar_ciclo_trading(self):
        """Execute a complete trading analysis and execution cycle"""
        print(f"\n{'='*60}")
        print(f"🔄 CICLO DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            # 1. Generate risk report
            self.risk_manager.generar_reporte_riesgo_avanzado()
            
            # 2. Verify conditions
            if not self.verificar_condiciones_trading():
                logger.info("Trading conditions not met, skipping cycle")
                return
            
            # 3. Execute multi-agent analysis
            print("\n🤖 INICIANDO ANÁLISIS MULTI-AGENTE...")
            logger.info("Starting multi-agent analysis")
            self.sistema_multiagente.execute_analysis(SYMBOLS)
            
            print(f"\n✅ Ciclo completado exitosamente")
            logger.info("Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            print(f"\n❌ Error en ciclo de trading: {e}")
    
    def generar_resumen_diario(self):
        """Generate daily trading summary"""
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DIARIO - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        try:
            # Get final metrics
            metricas = self.risk_manager.get_metricas_dia()
            
            print(f"\n📈 RESULTADOS DEL DÍA:")
            print(f"  • Total trades: {metricas['trades_totales']}")
            print(f"  • Ganados: {metricas['trades_ganadores']} | Perdidos: {metricas['trades_perdedores']}")
            print(f"  • Win Rate: {metricas['win_rate']*100:.0f}%")
            print(f"  • P&L: ${metricas['pnl_total']:.2f} ({metricas['pnl_porcentaje']:+.1f}%)")
            
            # Generate final risk report
            self.risk_manager.generar_reporte_riesgo_avanzado()
            
            logger.info(f"Daily summary generated - P&L: ${metricas['pnl_total']:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}", exc_info=True)
            print(f"\n❌ Error generando resumen: {e}")
    
    def modo_prueba(self):
        """Execute a single test cycle"""
        print("\n🧪 MODO PRUEBA - Ejecutando un ciclo...")
        logger.info("Starting test mode")
        self.ejecutar_ciclo_trading()
        print("\n✅ Prueba completada")
    
    def modo_continuo(self):
        """Execute continuous trading mode"""
        print("\n🏃 MODO CONTINUO - Trading automático activado")
        print("⏹️ Presiona Ctrl+C para detener")
        logger.info("Starting continuous trading mode")
        
        # Schedule tasks
        schedule.every(5).minutes.do(self.ejecutar_ciclo_trading)
        schedule.every().day.at("22:00").do(self.generar_resumen_diario)
        
        # Execute first cycle immediately
        self.ejecutar_ciclo_trading()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n⏹️ Sistema detenido por el usuario")
                logger.info("System stopped by user")
                self.generar_resumen_diario()
                break
            except Exception as e:
                logger.error(f"Error in continuous mode: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
                print("⏳ Reintentando en 60 segundos...")
                time.sleep(60)
    
    def modo_analisis_simple(self):
        """Execute analysis without real trades"""
        print("\n🔍 MODO ANÁLISIS - Solo análisis sin trades")
        logger.info("Starting analysis mode (no real trades)")
        
        # Temporarily disable trade execution
        original_execute = self.sistema_multiagente.ejecutar_trade_profesional
        self.sistema_multiagente.ejecutar_trade_profesional = \
            lambda x: print(f"  📝 [SIMULADO] {x['symbol']} {x['decision']}")
        
        try:
            self.ejecutar_ciclo_trading()
        finally:
            # Restore original function
            self.sistema_multiagente.ejecutar_trade_profesional = original_execute
    
    def ejecutar_health_check(self):
        """Execute system health check"""
        print("\n🏥 Ejecutando Health Check...")
        
        try:
            # Import and execute health_check directly
            import importlib.util
            health_check_path = UTILS_DIR / 'health_check.py'
            
            if not health_check_path.exists():
                print(f"❌ No se encontró {health_check_path}")
                return
            
            spec = importlib.util.spec_from_file_location("health_check", health_check_path)
            if spec and spec.loader:
                health_module = importlib.util.module_from_spec(spec)
                health_module.__file__ = str(health_check_path)
                
                # Add paths to sys.path temporarily
                import sys
                original_path = sys.path.copy()
                for path in [str(UTILS_DIR), str(PROJECT_ROOT), str(SRC_DIR)]:
                    if path not in sys.path:
                        sys.path.insert(0, path)
                
                try:
                    spec.loader.exec_module(health_module)
                    if hasattr(health_module, 'main'):
                        # Capturar SystemExit para evitar que termine el programa
                        try:
                            health_module.main()
                        except SystemExit:
                            pass  # Ignorar el exit del health check
                    else:
                        print("❌ health_check.py no tiene función main()")
                finally:
                    sys.path = original_path
            else:
                print("❌ No se pudo cargar el módulo health_check")
                
        except Exception as e:
            print(f"❌ Error al ejecutar health check: {e}")
            import traceback
            traceback.print_exc()
    
    def ejecutar_verify_system(self):
        """Execute complete system verification"""
        print("\n🔍 Ejecutando verificación del sistema...")
        
        try:
            # Import and execute verify_system directly
            import importlib.util
            verify_path = UTILS_DIR / 'verify_system.py'
            
            if not verify_path.exists():
                print(f"❌ No se encontró {verify_path}")
                return
            
            spec = importlib.util.spec_from_file_location("verify_system", verify_path)
            if spec and spec.loader:
                verify_module = importlib.util.module_from_spec(spec)
                verify_module.__file__ = str(verify_path)
                
                # Add paths to sys.path temporarily
                import sys
                original_path = sys.path.copy()
                for path in [str(UTILS_DIR), str(PROJECT_ROOT), str(SRC_DIR), str(SRC_DIR / 'strategies')]:
                    if path not in sys.path:
                        sys.path.insert(0, path)
                
                try:
                    spec.loader.exec_module(verify_module)
                    if hasattr(verify_module, 'main'):
                        # Capturar SystemExit para evitar que termine el programa
                        try:
                            verify_module.main()
                        except SystemExit:
                            pass  # Ignorar el exit del verify system
                    else:
                        print("❌ verify_system.py no tiene función main()")
                finally:
                    sys.path = original_path
            else:
                print("❌ No se pudo cargar el módulo verify_system")
                
        except Exception as e:
            print(f"❌ Error al ejecutar verify system: {e}")
            import traceback
            traceback.print_exc()
    
    def ejecutar_practice_mode(self):
        """Execute practice mode without restrictions"""
        print("\n🎮 Ejecutando modo práctica...")
        success = self.module_loader.execute_module_main(
            'practice_mode',
            SRC_DIR / 'practice_mode.py'
        )
        if not success:
            print("❌ No se pudo ejecutar el modo práctica")


class MenuManager:
    """Manages the interactive menu system"""
    
    @staticmethod
    def mostrar_menu() -> str:
        """
        Display main menu and get user selection.
        
        Returns:
            User's menu selection
        """
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


def main():
    """Main entry point for the trading bot"""
    try:
        # Initialize bot
        bot = TradingBotProfesional()
        menu = MenuManager()
        
        # Main menu loop
        while True:
            opcion = menu.mostrar_menu()
            
            actions = {
            "1": lambda: (bot.modo_prueba(), safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "2": lambda: bot.modo_continuo(),
                "3": lambda: (bot.modo_analisis_simple(), safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "4": lambda: (bot.ejecutar_health_check(), safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "5": lambda: (bot.ejecutar_verify_system(), safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "6": lambda: bot.ejecutar_practice_mode(),
                "7": lambda: (bot.risk_manager.generar_reporte_riesgo_avanzado(), 
                             safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "8": lambda: (bot.generar_resumen_diario(), safe_input("\nPresiona Enter para volver al menú (q para salir)...")),
                "9": lambda: (print("\n👋 ¡Hasta luego!"), exit(0))
            }
            
            action = actions.get(opcion)
            if action:
                action()
            else:
                print("\n❌ Opción no válida")
                
    except KeyboardInterrupt:
        print("\n\n⏹️ Sistema detenido por el usuario")
        logger.info("System stopped by user interrupt")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Error fatal: {e}")
        print("\n💡 Sugerencias:")
        print("   1. Verifica que todos los archivos __init__.py existan")
        print("   2. Asegúrate de estar ejecutando desde la carpeta correcta")
        print("   3. Revisa los logs para más detalles del error")


if __name__ == "__main__":
    main()