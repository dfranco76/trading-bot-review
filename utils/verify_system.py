#!/usr/bin/env python3
"""
System Verification Module for Trading Bot
Comprehensive testing of all trading system components.
"""

import sys
import os
from pathlib import Path
import logging
import importlib
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths properly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
STRATEGIES_DIR = SRC_DIR / 'strategies'

# Add all necessary paths
for path in [PROJECT_ROOT, SRC_DIR, STRATEGIES_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class ComponentVerifier:
    """Verifies individual system components"""
    
    def __init__(self):
        self.results = []
        self.test_data = {
            'symbol': 'AAPL',
            'period': '1d',
            'interval': '5m'
        }
    
    def verify_component(self, 
                        component_name: str, 
                        module_path: str, 
                        class_name: str,
                        test_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a single component.
        
        Args:
            component_name: Display name of the component
            module_path: Import path of the module
            class_name: Name of the class to instantiate
            test_method: Optional method to call for testing
            
        Returns:
            Dict with verification results
        """
        print(f"\n{'='*60}")
        print(f"PROBANDO: {component_name}")
        print("="*60)
        
        result = {
            'component': component_name,
            'status': 'unknown',
            'error': None,
            'details': {}
        }
        
        try:
            # Import module
            logger.info(f"Importing {module_path}...")
            module = importlib.import_module(module_path)
            
            # Get class
            if not hasattr(module, class_name):
                raise AttributeError(f"Module {module_path} has no class {class_name}")
            
            ComponentClass = getattr(module, class_name)
            
            # Instantiate
            logger.info(f"Creating instance of {class_name}...")
            instance = ComponentClass()
            
            # Basic verification
            result['details']['import'] = 'success'
            result['details']['instantiation'] = 'success'
            
            # Test specific method if provided
            if test_method and hasattr(instance, test_method):
                logger.info(f"Testing {test_method} method...")
                test_result = getattr(instance, test_method)(**self.test_data)
                result['details']['method_test'] = 'success'
                result['details']['test_output'] = str(test_result)[:100] + '...'
            
            result['status'] = 'pass'
            print(f"   ✅ {component_name} verificado correctamente")
            
        except ImportError as e:
            result['status'] = 'fail'
            result['error'] = f"Import error: {str(e)}"
            logger.error(f"Failed to import {module_path}: {e}")
            print(f"   ❌ Error de importación: {e}")
            
        except AttributeError as e:
            result['status'] = 'fail'
            result['error'] = f"Attribute error: {str(e)}"
            logger.error(f"Attribute error in {component_name}: {e}")
            print(f"   ❌ Error de atributo: {e}")
            
        except Exception as e:
            result['status'] = 'fail'
            result['error'] = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Unexpected error in {component_name}: {e}", exc_info=True)
            print(f"   ❌ Error: {e}")
            print(traceback.format_exc())
        
        self.results.append(result)
        return result


class SystemVerifier:
    """Main system verification orchestrator"""
    
    def __init__(self):
        self.component_verifier = ComponentVerifier()
        self.components_to_verify = [
            {
                'name': 'AgenteMomentum',
                'module': 'agente_momentum',
                'class': 'AgenteMomentum',
                'test_method': 'analyze'
            },
            {
                'name': 'AgenteMeanReversion',
                'module': 'agente_mean_reversion',
                'class': 'AgenteMeanReversion',
                'test_method': 'analyze'
            },
            {
                'name': 'AgentePatternRecognition',
                'module': 'agente_pattern_recognition',
                'class': 'AgentePatternRecognition',
                'test_method': 'analyze'
            },
            {
                'name': 'AgenteVolumeMomentum',
                'module': 'agente_volume_momentum',
                'class': 'AgenteVolumeMomentum',
                'test_method': 'analyze'
            },
            {
                'name': 'AgenteSentiment',
                'module': 'agente_sentiment',
                'class': 'AgenteSentiment',
                'test_method': 'analyze'
            },
            {
                'name': 'MLPredictionSystem',
                'module': 'ml_prediction_system',
                'class': 'MLPredictionSystem',
                'test_method': None  # No tiene analyze
            },
            {
                'name': 'PortfolioOptimizationSystem',
                'module': 'portfolio_optimization_system',
                'class': 'PortfolioOptimizationSystem',
                'test_method': None  # No tiene analyze
            },
            {
                'name': 'SistemaMultiAgente',
                'module': 'sistema_multiagente',
                'class': 'SistemaMultiAgente',
                'test_method': None  # Este coordina a todos
            }
        ]
        
        # Additional components to verify
        self.additional_components = [
            {
                'name': 'EnhancedRiskManager',
                'module': 'src.risk_management.risk_manager',
                'class': 'EnhancedRiskManager',
                'test_method': None
            },
            {
                'name': 'ClaudeMentor',
                'module': 'src.utils.claude_mentor',
                'class': 'ClaudeMentor',
                'test_method': None
            }
        ]
    
    def verify_imports(self) -> Dict[str, bool]:
        """Verify basic imports work correctly"""
        print("\n🔍 Verificando imports básicos...")
        
        imports_to_check = [
            ('yfinance', 'yfinance'),
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('ta', 'ta'),
            ('sklearn', 'scikit-learn'),
            ('schedule', 'schedule')
        ]
        
        results = {}
        for module_name, display_name in imports_to_check:
            try:
                importlib.import_module(module_name)
                results[display_name] = True
                print(f"   ✅ {display_name}")
            except ImportError:
                results[display_name] = False
                print(f"   ❌ {display_name}")
        
        return results
    
    def verify_file_structure(self) -> Dict[str, List[str]]:
        """Verify project file structure"""
        print("\n📁 Verificando estructura de archivos...")
        
        required_files = {
            'strategies': [
                'agente_momentum.py',
                'agente_mean_reversion.py',
                'agente_pattern_recognition.py',
                'agente_volume_momentum.py',
                'agente_sentiment.py',
                'sistema_multiagente.py'
            ],
            'risk_management': [
                'risk_manager.py',
                '__init__.py'
            ],
            'utils': [
                'config.py',
                'claude_mentor.py',
                '__init__.py'
            ]
        }
        
        missing_files = {}
        
        for directory, files in required_files.items():
            dir_path = SRC_DIR / directory
            missing = []
            
            for file in files:
                file_path = dir_path / file
                if not file_path.exists():
                    missing.append(file)
            
            if missing:
                missing_files[directory] = missing
                print(f"   ❌ {directory}: Faltan {len(missing)} archivos")
            else:
                print(f"   ✅ {directory}: Todos los archivos presentes")
        
        return missing_files
    
    def run_verification(self) -> Dict[str, Any]:
        """Run complete system verification"""
        
        print("\n" + "="*60)
        print("🔧 VERIFICACIÓN COMPLETA DEL SISTEMA")
        print("="*60)
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Directorio: {PROJECT_ROOT}")
        
        # Step 1: Verify imports
        import_results = self.verify_imports()
        
        # Step 2: Verify file structure
        missing_files = self.verify_file_structure()
        
        # Step 3: Verify trading components
        print("\n🤖 Verificando componentes de trading...")
        
        for component in self.components_to_verify:
            self.component_verifier.verify_component(
                component_name=component['name'],
                module_path=component['module'],
                class_name=component['class'],
                test_method=component.get('test_method')
            )
        
        # Step 4: Verify additional components
        print("\n🔧 Verificando componentes adicionales...")
        
        for component in self.additional_components:
            self.component_verifier.verify_component(
                component_name=component['name'],
                module_path=component['module'],
                class_name=component['class'],
                test_method=component.get('test_method')
            )
        
        # Compile results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(PROJECT_ROOT),
            'imports': import_results,
            'missing_files': missing_files,
            'components': self.component_verifier.results
        }
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print verification summary"""
        
        print("\n" + "="*60)
        print("📊 RESUMEN DE RESULTADOS")
        print("="*60)
        
        # Import summary
        imports_ok = sum(1 for ok in results['imports'].values() if ok)
        imports_total = len(results['imports'])
        print(f"\n📦 Dependencias: {imports_ok}/{imports_total} instaladas")
        
        # File structure summary
        if results['missing_files']:
            print(f"\n📁 Archivos faltantes en:")
            for directory, files in results['missing_files'].items():
                print(f"   • {directory}: {', '.join(files)}")
        else:
            print("\n📁 Estructura de archivos: ✅ Completa")
        
        # Component summary
        components_ok = sum(1 for comp in results['components'] 
                          if comp['status'] == 'pass')
        components_total = len(results['components'])
        
        print(f"\n🤖 Componentes verificados:")
        for component in results['components']:
            icon = '✅' if component['status'] == 'pass' else '❌'
            print(f"{icon} {component['component']}: {'FUNCIONANDO' if component['status'] == 'pass' else 'CON ERRORES'}")
        
        print(f"\n📈 Total: {components_ok}/{components_total} componentes funcionando")
        
        # Overall status
        if components_ok == components_total and not results['missing_files']:
            print("\n💚 SISTEMA COMPLETAMENTE OPERATIVO")
        elif components_ok > components_total * 0.7:
            print("\n💛 SISTEMA PARCIALMENTE OPERATIVO")
            print("⚠️ Algunos componentes necesitan atención")
        else:
            print("\n❤️ SISTEMA NO OPERATIVO")
            print("⚠️ HAY COMPONENTES CON ERRORES")
            print("💡 Revisa los errores anteriores y corrige los archivos necesarios")
    
    def save_report(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save verification report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_verification_report_{timestamp}.json"
        
        report_path = PROJECT_ROOT / 'logs' / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Reporte guardado en: {report_path}")


def main():
    """Main entry point for system verification"""
    try:
        verifier = SystemVerifier()
        results = verifier.run_verification()
        
        # Save report
        verifier.save_report(results)
        
        # Return appropriate exit code
        components_ok = sum(1 for comp in results['components'] 
                          if comp['status'] == 'pass')
        if components_ok == len(results['components']):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in system verification: {e}", exc_info=True)
        print(f"\n❌ Error fatal en verificación: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()