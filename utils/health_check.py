# utils/health_check.py
# Script para verificar la salud del sistema antes de operar

import sys
sys.path.insert(0, 'src')

from datetime import datetime
from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
import yfinance as yf

def check_database_connection():
    """Verifica conexión a Supabase"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        response = supabase.table('bot_status').select("*").execute()
        return True, f"✅ Conexión a BD exitosa"
    except Exception as e:
        return False, f"❌ Error de BD: {e}"

def check_market_data():
    """Verifica acceso a datos de mercado"""
    try:
        stock = yf.Ticker("AAPL")
        data = stock.history(period="1d")
        if len(data) > 0:
            return True, f"✅ Datos de mercado disponibles"
        else:
            return False, "❌ No hay datos de mercado"
    except Exception as e:
        return False, f"❌ Error con yfinance: {e}"

def check_trading_hours():
    """Verifica si es horario de trading"""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    
    if weekday >= 5:
        return False, "⏸️ Fin de semana - Mercado cerrado"
    elif hour < 15 or hour >= 22:
        return False, f"⏸️ Fuera de horario (actual: {hour}h)"
    else:
        return True, "✅ Horario de trading activo"

def check_risk_limits():
    """Verifica límites de riesgo"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Verificar capital
        response = supabase.table('bot_status').select("capital").execute()
        capital = float(response.data[0]['capital']) if response.data else 0
        
        # Verificar exposición
        trades = supabase.table('trades').select("*").eq('status', 'OPEN').execute()
        exposicion = sum(t['quantity'] * t['price'] for t in trades.data)
        exposicion_pct = (exposicion / capital * 100) if capital > 0 else 0
        
        if exposicion_pct > 80:
            return False, f"⚠️ Exposición muy alta: {exposicion_pct:.1f}%"
        else:
            return True, f"✅ Exposición OK: {exposicion_pct:.1f}%"
            
    except Exception as e:
        return False, f"❌ Error verificando riesgo: {e}"

def main():
    print("\n" + "="*50)
    print("🏥 HEALTH CHECK - SISTEMA DE TRADING")
    print("="*50)
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Base de Datos", check_database_connection),
        ("Datos de Mercado", check_market_data),
        ("Horario Trading", check_trading_hours),
        ("Límites de Riesgo", check_risk_limits)
    ]
    
    all_good = True
    
    for name, check_func in checks:
        print(f"\n🔍 Verificando {name}...")
        success, message = check_func()
        print(f"   {message}")
        if not success:
            all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✅ SISTEMA LISTO PARA OPERAR")
    else:
        print("⚠️ HAY PROBLEMAS - REVISAR ANTES DE OPERAR")
    print("="*50)

if __name__ == "__main__":
    main()
    input("\nPresiona Enter para salir...")