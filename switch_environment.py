#!/usr/bin/env python3
"""
Script para cambiar entre modo desarrollo y producción
Uso: python switch_environment.py [dev|prod|status]
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Colores para la terminal
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(text, color=Colors.WHITE):
    """Imprime texto con color"""
    print(f"{color}{text}{Colors.RESET}")

def get_current_environment():
    """Lee el entorno actual del archivo .env"""
    env_file = Path('.env')
    if not env_file.exists():
        return None
    
    with open(env_file, 'r') as f:
        for line in f:
            if line.strip().startswith('ENVIRONMENT='):
                return line.split('=')[1].strip()
    return None

def show_status():
    """Muestra el estado actual del entorno"""
    print_colored("\n📊 ESTADO ACTUAL DEL ENTORNO", Colors.CYAN)
    print("=" * 60)
    
    current_env = get_current_environment()
    
    if current_env == 'production':
        print_colored("🔴 Modo: PRODUCCIÓN", Colors.RED)
        print("   • Trades REALES")
        print("   • Validaciones estrictas")
        print("   • Reportes automáticos")
        print("   • Logs completos")
    elif current_env == 'development':
        print_colored("🟢 Modo: DESARROLLO", Colors.GREEN)
        print("   • Trades simulados disponibles")
        print("   • Validaciones relajadas")
        print("   • Reportes opcionales")
        print("   • Debug activado")
    else:
        print_colored("⚠️  Modo: NO CONFIGURADO", Colors.YELLOW)
        print("   Ejecuta: python switch_environment.py dev")

def preserve_api_keys():
    """Preserva las API keys del archivo .env actual"""
    env_file = Path('.env')
    preserved = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if key in ['SUPABASE_URL', 'SUPABASE_KEY', 'ALPHA_VANTAGE_API_KEY', 'YFINANCE_API_KEY']:
                        if value:  # Solo preservar si tiene valor
                            preserved[key] = value
    
    return preserved

def set_development():
    """Configura el entorno de desarrollo"""
    print_colored("\n🔄 Cambiando a modo DESARROLLO...", Colors.YELLOW)
    
    # Preservar API keys existentes
    preserved = preserve_api_keys()
    
    env_content = f"""# Trading Bot Environment Variables - MODO DESARROLLO
ENVIRONMENT=development

# Database Configuration (opcional en desarrollo)
SUPABASE_URL={preserved.get('SUPABASE_URL', '')}
SUPABASE_KEY={preserved.get('SUPABASE_KEY', '')}

# API Keys (opcional)
ALPHA_VANTAGE_API_KEY={preserved.get('ALPHA_VANTAGE_API_KEY', '')}
YFINANCE_API_KEY={preserved.get('YFINANCE_API_KEY', '')}

# Trading Configuration
DEFAULT_CAPITAL=10000
MAX_RISK_PER_TRADE=0.02

# Logging
LOG_LEVEL=DEBUG

# Development Settings
SAVE_REPORTS=false
STRICT_VALIDATION=false
ALLOW_PAPER_TRADING=true
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Crear configuración adicional
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config = {
        "environment": "development",
        "capital": 10000,
        "max_risk_per_trade": 0.02,
        "paper_trading": True,
        "save_reports": False,
        "features": {
            "use_real_money": False,
            "enable_notifications": False,
            "strict_validation": False,
            "auto_trading": False
        }
    }
    
    with open(config_dir / 'environment.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print_colored("✅ Modo DESARROLLO activado", Colors.GREEN)
    print("\n💡 Características del modo desarrollo:")
    print("   • Puedes usar Paper Trading (simulado)")
    print("   • No se requiere SUPABASE_URL")
    print("   • Logs en modo DEBUG")
    print("   • Sin validaciones estrictas")

def set_production():
    """Configura el entorno de producción"""
    print_colored("\n🔄 Cambiando a modo PRODUCCIÓN...", Colors.YELLOW)
    print_colored("⚠️  ADVERTENCIA: En modo producción los trades son REALES", Colors.RED)
    
    confirm = input("¿Estás seguro? (S/N): ")
    if confirm.lower() != 's':
        print_colored("Operación cancelada", Colors.YELLOW)
        return
    
    # Preservar API keys existentes
    preserved = preserve_api_keys()
    
    env_content = f"""# Trading Bot Environment Variables - MODO PRODUCCIÓN
ENVIRONMENT=production

# Database Configuration (REQUERIDO en producción)
SUPABASE_URL={preserved.get('SUPABASE_URL', '')}
SUPABASE_KEY={preserved.get('SUPABASE_KEY', '')}

# API Keys (RECOMENDADO)
ALPHA_VANTAGE_API_KEY={preserved.get('ALPHA_VANTAGE_API_KEY', '')}
YFINANCE_API_KEY={preserved.get('YFINANCE_API_KEY', '')}

# Trading Configuration
DEFAULT_CAPITAL=10000
MAX_RISK_PER_TRADE=0.02

# Logging
LOG_LEVEL=INFO

# Production Settings
SAVE_REPORTS=true
STRICT_VALIDATION=true
ALLOW_PAPER_TRADING=false
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Crear configuración adicional
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config = {
        "environment": "production",
        "capital": 10000,
        "max_risk_per_trade": 0.02,
        "paper_trading": False,
        "save_reports": True,
        "features": {
            "use_real_money": True,
            "enable_notifications": True,
            "strict_validation": True,
            "auto_trading": True
        }
    }
    
    with open(config_dir / 'environment.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Crear directorios necesarios
    for directory in ['logs', 'logs/reports', 'data', 'results']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_colored("✅ Modo PRODUCCIÓN activado", Colors.RED)
    print("\n⚠️  Características del modo producción:")
    print("   • Trades con dinero REAL")
    print("   • Se requiere configurar SUPABASE")
    print("   • Reportes automáticos en logs/")
    print("   • Validaciones estrictas activadas")

def main():
    """Función principal"""
    args = sys.argv[1:] if len(sys.argv) > 1 else ['status']
    mode = args[0].lower() if args else 'status'
    
    if mode in ['dev', 'development']:
        set_development()
        show_status()
    elif mode in ['prod', 'production']:
        set_production()
        show_status()
    elif mode == 'status':
        show_status()
    else:
        print_colored(f"❌ Modo no reconocido: {mode}", Colors.RED)
        print("\nUso: python switch_environment.py [dev|prod|status]")
        return
    
    print_colored("\n🚀 Comandos disponibles:", Colors.CYAN)
    print("   python switch_environment.py dev     # Cambiar a desarrollo")
    print("   python switch_environment.py prod    # Cambiar a producción")
    print("   python switch_environment.py         # Ver estado actual\n")

if __name__ == "__main__":
    main()