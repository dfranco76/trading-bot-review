# src/trade_manager.py
"""
Sistema de gestión activa de trades con stop loss y take profit automáticos
"""
from config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
import yfinance as yf
from datetime import datetime
import time
import threading

class TradeManager:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.active_trades = {}
        self.monitoring = False
        
    def add_trade_with_stops(self, trade_info):
        """Añade un trade con stop loss y take profit"""
        trade_id = trade_info['id']
        
        # Calcular niveles de salida
        if trade_info['action'] == 'BUY':
            stop_loss = trade_info['price'] * 0.98  # -2%
            take_profit = trade_info['price'] * 1.03  # +3%
        else:  # SELL
            stop_loss = trade_info['price'] * 1.02  # +2%
            take_profit = trade_info['price'] * 0.97  # -3%
        
        self.active_trades[trade_id] = {
            'symbol': trade_info['symbol'],
            'action': trade_info['action'],
            'entry_price': trade_info['price'],
            'quantity': trade_info['quantity'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': False,
            'max_price': trade_info['price']  # Para trailing stop
        }
        
        # Actualizar en BD
        self.supabase.table('trades').update({
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'MONITORED'
        }).eq('id', trade_id).execute()
        
        print(f"📊 Trade {trade_id} añadido con SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
    
    def activate_trailing_stop(self, trade_id, trailing_percent=0.01):
        """Activa trailing stop loss"""
        if trade_id in self.active_trades:
            self.active_trades[trade_id]['trailing_stop'] = True
            self.active_trades[trade_id]['trailing_percent'] = trailing_percent
            print(f"🎯 Trailing stop activado para trade {trade_id}")
    
    def monitor_trades(self):
        """Monitorea todos los trades activos"""
        self.monitoring = True
        
        while self.monitoring:
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    # Obtener precio actual
                    ticker = yf.Ticker(trade['symbol'])
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    
                    # Actualizar max price para trailing stop
                    if trade['action'] == 'BUY' and current_price > trade['max_price']:
                        trade['max_price'] = current_price
                        
                        # Ajustar trailing stop
                        if trade['trailing_stop']:
                            new_stop = current_price * (1 - trade['trailing_percent'])
                            trade['stop_loss'] = max(trade['stop_loss'], new_stop)
                            print(f"📈 Trailing stop actualizado para {trade['symbol']}: ${new_stop:.2f}")
                    
                    # Verificar condiciones de salida
                    should_exit = False
                    exit_reason = ""
                    
                    if trade['action'] == 'BUY':
                        if current_price <= trade['stop_loss']:
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price >= trade['take_profit']:
                            should_exit = True
                            exit_reason = "Take Profit"
                    else:  # SELL
                        if current_price >= trade['stop_loss']:
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price <= trade['take_profit']:
                            should_exit = True
                            exit_reason = "Take Profit"
                    
                    if should_exit:
                        self.execute_exit(trade_id, current_price, exit_reason)
                        
                except Exception as e:
                    print(f"Error monitoreando {trade['symbol']}: {e}")
            
            time.sleep(60)  # Verificar cada minuto
    
    def execute_exit(self, trade_id, exit_price, reason):
        """Ejecuta la salida de un trade"""
        trade = self.active_trades[trade_id]
        
        # Calcular P&L
        if trade['action'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
            pnl_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
            pnl_pct = ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
        
        # Actualizar en BD
        self.supabase.table('trades').update({
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl': pnl,
            'pnl_percentage': pnl_pct,
            'status': 'CLOSED',
            'closed_at': datetime.now().isoformat()
        }).eq('id', trade_id).execute()
        
        # Remover de activos
        del self.active_trades[trade_id]
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} Trade {trade_id} cerrado por {reason}")
        print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    
    def start_monitoring(self):
        """Inicia el monitoreo en un thread separado"""
        monitor_thread = threading.Thread(target=self.monitor_trades)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("🔍 Sistema de monitoreo de trades iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        print("⏹️ Sistema de monitoreo detenido")