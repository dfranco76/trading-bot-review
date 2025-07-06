"""
Script para verificar que las importaciones funcionan correctamente.
Ejecutar desde la raíz del proyecto o desde la carpeta tests.
"""
import sys
import os

print("=== VERIFICACIÓN DE IMPORTACIÓN ===")
print(f"Directorio actual: {os.getcwd()}")
print(f"Script ubicado en: {os.path.abspath(__file__)}")

# Buscar el directorio src
possible_paths = [
    os.path.join(os.getcwd(), 'src'),
    os.path.join(os.path.dirname(os.getcwd()), 'src'),
    os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src'),
    r'C:\Users\david\Repository\trading-bot\src',
]

src_found = False
for path in possible_paths:
    print(f"\nProbando path: {path}")
    if os.path.exists(path):
        print(f"  ✓ El directorio existe")
        risk_manager_path = os.path.join(path, 'risk_manager.py')
        if os.path.exists(risk_manager_path):
            print(f"  ✓ risk_manager.py encontrado")
            sys.path.insert(0, path)
            src_found = True
            break
        else:
            print(f"  ✗ risk_manager.py NO encontrado")
    else:
        print(f"  ✗ El directorio NO existe")

if not src_found:
    print("\n✗ ERROR: No se pudo encontrar el directorio src con risk_manager.py")
    sys.exit(1)

# Intentar importar
print("\n=== INTENTANDO IMPORTAR ===")
try:
    from risk_manager import RiskManager
    print("✓ Importación exitosa!")
    print(f"  RiskManager es una clase: {type(RiskManager)}")
    
    # Verificar que podemos crear una instancia
    rm = RiskManager()
    print(f"  Instancia creada: {rm}")
    print(f"  Capital inicial: {rm.capital_inicial}")
    
except ImportError as e:
    print(f"✗ Error de importación: {e}")
except Exception as e:
    print(f"✗ Error al crear instancia: {e}")

print("\n=== FIN DE VERIFICACIÓN ===")