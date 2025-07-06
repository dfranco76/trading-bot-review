# src/broker_integration.py
import alpaca_trade_api as tradeapi
from utils.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from datetime import datetime
import time

class BrokerIntegration:
    def __init__(self):
        """Inicializa conexión con el broker"""
        self.api = None
        self.paper_trading = "paper" in ALPACA_BASE_URL
        
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                self.api = tradeapi.REST(
                    ALPACA_API_KEY,
                    ALPACA_SECRET_KEY,
                    ALPACA_BASE_URL,
                    api_version='v2'
                )
                
                # Verificar conexión
                account = self.api.get_account()
                print(f"✅ Conectado a Alpaca ({'Paper' if self.paper_trading else 'Real'})")
                print(f"   Balance: ${float(account.equity):,.2f}")
                print(f"   Buying Power: ${float(account.buying_power):,.2f}")
                
            except Exception as e:
                print(f"❌ Error conectando con Alpaca: {e}")
        else:
            print("⚠️ Alpaca no configurado - Modo simulación")
    
    def get_account_info(self):
        """Obtiene información de la cuenta"""
        if not self.api:
            return None
            
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'positions_value': float(account.long_market_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            print(f"Error obteniendo info de cuenta: {e}")
            return None
    
    def get_positions(self):
        """Obtiene posiciones actuales"""
        if not self.api:
            return []
            
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc)
            } for pos in positions]
        except Exception as e:
            print(f"Error obteniendo posiciones: {e}")
            return []
    
    def place_order(self, symbol, qty, side, order_type='market', limit_price=None):
        """Coloca una orden"""
        if not self.api:
            print(f"📝 [SIMULADO] {side.upper()} {qty} {symbol}")
            return None
            
        try:
            # Verificar si el mercado está abierto
            clock = self.api.get_clock()
            if not clock.is_open:
                print(f"⚠️ Mercado cerrado - Orden guardada para próxima apertura")
            
            # Crear orden
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=limit_price
            )
            
            print(f"✅ Orden enviada: {side.upper()} {qty} {symbol}")
            print(f"   Order ID: {order.id}")
            
            return order
            
        except Exception as e:
            print(f"❌ Error enviando orden: {e}")
            return None
    
    def cancel_order(self, order_id):
        """Cancela una orden"""
        if not self.api:
            return False
            
        try:
            self.api.cancel_order(order_id)
            print(f"✅ Orden {order_id} cancelada")
            return True
        except Exception as e:
            print(f"Error cancelando orden: {e}")
            return False
    
    def get_market_hours(self):
        """Obtiene horario del mercado"""
        if not self.api:
            return None
            
        try:
            clock = self.api.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except:
            return None
    
    def execute_trade_from_signal(self, trade_signal):
        """Ejecuta un trade basado en señal del sistema"""
        symbol = trade_signal['symbol']
        action = trade_signal['action']
        quantity = int(trade_signal['quantity'])  # Alpaca requiere enteros
        
        # Verificar si ya tenemos posición
        positions = self.get_positions()
        current_position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if action == 'BUY':
            # Comprar
            order = self.place_order(symbol, quantity, 'buy')
            
        elif action == 'SELL':
            if current_position:
                # Si tenemos posición, vender todo
                order = self.place_order(symbol, current_position['qty'], 'sell')
            else:
                # Si no tenemos posición, es un short (cuidado!)
                print(f"⚠️ Intento de SELL sin posición en {symbol}")
                order = None
        
        return order


# Función para integrar con el sistema principal
def ejecutar_trade_real(trade_decision):
    """Integración con el sistema de trading"""
    broker = BrokerIntegration()
    
    # Verificar cuenta
    account = broker.get_account_info()
    if not account:
        print("❌ No se pudo conectar con el broker")
        return False
    
    if account['trading_blocked']:
        print("❌ Trading bloqueado en la cuenta")
        return False
    
    # Ejecutar trade
    order = broker.execute_trade_from_signal(trade_decision)
    
    if order:
        print(f"✅ Trade ejecutado en {'paper' if broker.paper_trading else 'real'} trading")
        return True
    else:
        print("❌ No se pudo ejecutar el trade")
        return False


if __name__ == "__main__":
    # Test de conexión
    print("🧪 TEST DE CONEXIÓN CON BROKER")
    print("="*50)
    
    broker = BrokerIntegration()
    
    if broker.api:
        print("\n📊 Información de cuenta:")
        info = broker.get_account_info()
        if info:
            for key, value in info.items():
                print(f"   {key}: {value}")
        
        print("\n📈 Posiciones actuales:")
        positions = broker.get_positions()
        if positions:
            for pos in positions:
                print(f"   {pos['symbol']}: {pos['qty']} @ ${pos['avg_price']:.2f} (P&L: {pos['unrealized_plpc']:.2f}%)")
        else:
            print("   Sin posiciones abiertas")
        
        print("\n⏰ Horario del mercado:")
        hours = broker.get_market_hours()
        if hours:
            print(f"   Mercado {'ABIERTO' if hours['is_open'] else 'CERRADO'}")
    else:
        print("\n⚠️ Para configurar Alpaca:")
        print("1. Crea cuenta en https://alpaca.markets/")
        print("2. Obtén tus API keys")
        print("3. Añádelas a tu archivo .env")
        print("4. Ejecuta este script de nuevo")