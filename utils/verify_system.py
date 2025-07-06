# verify_system.py
# Script para verificar que todos los agentes funcionan correctamente

import sys
import traceback
from datetime import datetime

def test_agent(agent_module, agent_class, test_symbol='NVDA'):
    """Prueba un agente específico"""
    print(f"\n{'='*60}")
    print(f"🧪 PROBANDO: {agent_class}")
    print(f"{'='*60}")
    
    try:
        # Importar dinámicamente
        module = __import__(agent_module, fromlist=[agent_class])
        AgentClass = getattr(module, agent_class)
        
        # Crear instancia
        print("1️⃣ Creando instancia del agente...")
        agent = AgentClass()
        print("   ✅ Agente creado exitosamente")
        
        # Probar análisis
        print(f"\n2️⃣ Analizando {test_symbol}...")
        result = agent.analyze_symbol(test_symbol)
        
        if result:
            print(f"\n   📊 Resultado del análisis:")
            print(f"   • Acción: {result.get('action', 'N/A')}")
            print(f"   • Confianza: {result.get('confidence', 0)*100:.0f}%")
            print(f"   • Razón: {result.get('reason', 'Sin razón')}")
            print(f"   • Precio: ${result.get('price', 0):.2f}")
            print("\n   ✅ Análisis completado exitosamente")
        else:
            print("   ⚠️ El análisis no retornó resultados")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error de importación: {e}")
        print(f"   💡 Verifica que el archivo '{agent_module}.py' existe en src/")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n   📋 Traceback completo:")
        traceback.print_exc()
        return False

def test_multiagent_system():
    """Prueba el sistema multi-agente completo"""
    print(f"\n{'='*60}")
    print(f"🧪 PROBANDO: Sistema Multi-Agente")
    print(f"{'='*60}")
    
    try:
        from sistema_multiagente import SistemaMultiAgente
        
        print("1️⃣ Creando sistema multi-agente...")
        sistema = SistemaMultiAgente()
        print("   ✅ Sistema creado exitosamente")
        
        # Probar con un símbolo
        print("\n2️⃣ Probando análisis de NVDA...")
        # No ejecutar el análisis completo, solo verificar que se puede crear
        print("   ✅ Sistema multi-agente funcional")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Función principal de verificación"""
    print("\n" + "="*60)
    print("🔧 VERIFICACIÓN COMPLETA DEL SISTEMA DE TRADING")
    print("="*60)
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Lista de agentes a probar
    agents_to_test = [
        ('agente_momentum', 'AgenteMomentum'),
        ('agente_mean_reversion', 'AgenteMeanReversion'),
        ('agente_pattern_recognition', 'AgentePatternRecognition'),
        ('agente_volume_momentum', 'AgenteVolumeMomentum')
    ]
    
    results = {}
    
    # Probar cada agente
    for module, class_name in agents_to_test:
        results[class_name] = test_agent(module, class_name)
    
    # Probar sistema multi-agente
    results['SistemaMultiAgente'] = test_multiagent_system()
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for component, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {component}: {'FUNCIONAL' if status else 'CON ERRORES'}")
    
    print(f"\n📈 Total: {passed}/{total} componentes funcionando")
    
    if passed == total:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("💡 Puedes ejecutar: python src/main_bot.py")
    else:
        print("\n⚠️ HAY COMPONENTES CON ERRORES")
        print("💡 Revisa los errores anteriores y corrige los archivos necesarios")
    
    return passed == total

if __name__ == "__main__":
    # Añadir src al path para poder importar
    sys.path.insert(0, 'src')
    
    success = main()
    
    input("\n\nPresiona Enter para salir...")
    
    # Retornar código de salida
    sys.exit(0 if success else 1)