# monitor.py - Script de monitoreo para trading real
import sys
sys.path.insert(0, 'src')

from utils.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class TradingMonitor:
    def __init__(self):
        self.initial_capital = 200.0
        
    def get_todays_trades(self):
        """Obtiene trades del día"""
        today = datetime.now().strftime('%Y-%m-%d')
        response = supabase.table('trades').select("*").gte('created_at', today).execute()
        return response.data
    
    def get_current_positions(self):
        """Obtiene posiciones abiertas"""
        response = supabase.table('trades').select("*").eq('status', 'OPEN').execute()
        return response.data
    
    def calculate_daily_pnl(self):
        """Calcula P&L del día"""
        trades = self.get_todays_trades()
        total_pnl = 0
        
        for trade in trades:
            if 'pnl' in trade and trade['pnl']:
                total_pnl += trade['pnl']
        
        return total_pnl
    
    def calculate_exposure(self):
        """Calcula exposición actual"""
        positions = self.get_current_positions()
        total_exposure = sum(p['quantity'] * p['price'] for p in positions)
        exposure_pct = (total_exposure / self.initial_capital) * 100
        return total_exposure, exposure_pct
    
    def get_performance_metrics(self):
        """Calcula métricas de performance"""
        # Últimos 7 días
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        response = supabase.table('trades').select("*").gte('created_at', week_ago).execute()
        trades = response.data
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def check_risk_limits(self):
        """Verifica límites de riesgo"""
        warnings = []
        
        # Check daily loss
        daily_pnl = self.calculate_daily_pnl()
        if daily_pnl <= -10:
            warnings.append("🚨 LÍMITE DIARIO ALCANZADO: Pérdida > 10€")
        elif daily_pnl <= -5:
            warnings.append("⚠️ Precaución: Pérdida > 5€")
        
        # Check exposure
        _, exposure_pct = self.calculate_exposure()
        if exposure_pct > 80:
            warnings.append("🚨 EXPOSICIÓN MUY ALTA: > 80%")
        elif exposure_pct > 60:
            warnings.append("⚠️ Exposición alta: > 60%")
        
        # Check número de trades
        todays_trades = self.get_todays_trades()
        if len(todays_trades) >= 5:
            warnings.append("⚠️ Muchos trades hoy: Considerar parar")
        
        return warnings
    
    def generate_report(self):
        """Genera reporte completo"""
        print("\n" + "="*60)
        print(f"📊 MONITOR DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Capital y P&L
        current_capital = supabase.table('bot_status').select("capital").execute().data[0]['capital']
        daily_pnl = self.calculate_daily_pnl()
        daily_pnl_pct = (daily_pnl / self.initial_capital) * 100
        
        print(f"\n💰 CAPITAL:")
        print(f"  • Inicial: ${self.initial_capital:.2f}")
        print(f"  • Actual: ${current_capital:.2f}")
        print(f"  • P&L Hoy: ${daily_pnl:.2f} ({daily_pnl_pct:+.1f}%)")
        
        # Exposición
        total_exp, exp_pct = self.calculate_exposure()
        print(f"\n📊 EXPOSICIÓN:")
        print(f"  • Total: ${total_exp:.2f} ({exp_pct:.1f}%)")
        
        # Posiciones abiertas
        positions = self.get_current_positions()
        if positions:
            print(f"\n📈 POSICIONES ABIERTAS ({len(positions)}):")
            for pos in positions:
                print(f"  • {pos['symbol']}: {pos['quantity']:.2f} @ ${pos['price']:.2f}")
        
        # Trades del día
        todays_trades = self.get_todays_trades()
        print(f"\n📝 TRADES HOY ({len(todays_trades)}):")
        if todays_trades:
            df = pd.DataFrame(todays_trades)[['created_at', 'symbol', 'action', 'price', 'quantity', 'pnl']]
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%H:%M')
            print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))
        
        # Performance
        metrics = self.get_performance_metrics()
        print(f"\n📈 PERFORMANCE (7 días):")
        print(f"  • Total trades: {metrics['total_trades']}")
        print(f"  • Win rate: {metrics['win_rate']*100:.0f}%")
        print(f"  • Profit factor: {metrics['profit_factor']:.2f}")
        
        # Warnings
        warnings = self.check_risk_limits()
        if warnings:
            print(f"\n⚠️ ALERTAS:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print(f"\n✅ Sin alertas - Todo dentro de límites")
        
        print("\n" + "="*60)
    
    def monitor_live(self, interval_seconds=60):
        """Monitoreo en vivo"""
        import time
        
        print("🔄 Iniciando monitoreo en vivo...")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                self.generate_report()
                
                # Check crítico
                warnings = self.check_risk_limits()
                if any("LÍMITE DIARIO ALCANZADO" in w for w in warnings):
                    print("\n🛑 DETENIENDO BOT - LÍMITE DIARIO ALCANZADO")
                    break
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoreo detenido")


def quick_check():
    """Chequeo rápido de estado"""
    monitor = TradingMonitor()
    
    daily_pnl = monitor.calculate_daily_pnl()
    _, exposure_pct = monitor.calculate_exposure()
    warnings = monitor.check_risk_limits()
    
    status = "🟢 OK" if not warnings else "🟡 ALERTA" if len(warnings) < 2 else "🔴 CRÍTICO"
    
    print(f"\n{status} | P&L: ${daily_pnl:.2f} | Exp: {exposure_pct:.0f}% | Trades: {len(monitor.get_todays_trades())}")
    
    if warnings:
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        # Monitoreo en vivo
        monitor = TradingMonitor()
        monitor.monitor_live(30)  # Actualizar cada 30 segundos
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Chequeo rápido
        quick_check()
    else:
        # Reporte completo
        monitor = TradingMonitor()
        monitor.generate_report()