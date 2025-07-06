# pro_trading_system.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProTradingSystem:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = 0.01  # Solo arriesgar 1% por operación
        self.positions = {}
        self.performance = []
        
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calcula el tamaño óptimo de posición usando Kelly Criterion"""
        if avg_loss == 0:
            return 0
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        # Usar 25% del Kelly para ser conservador
        return max(0, min(kelly * 0.25, 0.25))
    
    def find_momentum_stocks(self):
        """Encuentra las acciones con mejor momentum"""
        print("\n🔍 Buscando las mejores oportunidades...")
        
        # Lista de acciones de alta liquidez
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
                  'AMD', 'NFLX', 'ADBE', 'CRM', 'PYPL', 'SQ', 'SHOP', 'COIN']
        
        momentum_scores = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                
                if len(hist) > 0:
                    # Calcular momentum (retorno de 1 mes)
                    returns_1m = (hist['Close'][-20:].iloc[-1] / hist['Close'][-20:].iloc[0] - 1) * 100
                    # Volatilidad
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                    # Score: momentum / volatilidad
                    score = returns_1m / volatility if volatility > 0 else 0
                    
                    momentum_scores[symbol] = {
                        'returns_1m': returns_1m,
                        'volatility': volatility,
                        'score': score,
                        'price': hist['Close'].iloc[-1]
                    }
            except:
                continue
        
        # Top 5 acciones por momentum
        top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
        
        print("\n📈 TOP 5 ACCIONES CON MEJOR MOMENTUM:")
        print("-" * 60)
        for symbol, data in top_stocks:
            print(f"{symbol}: Retorno 1M: {data['returns_1m']:.1f}%, "
                  f"Volatilidad: {data['volatility']:.1f}%, "
                  f"Score: {data['score']:.2f}")
        
        return [stock[0] for stock in top_stocks]
    
    def advanced_strategy(self, symbol, period="6mo"):
        """Estrategia avanzada multi-factor"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # Indicadores
        # 1. Trend Following
        data['EMA_20'] = data['Close'].ewm(span=20).mean()
        data['EMA_50'] = data['Close'].ewm(span=50).mean()
        
        # 2. Mean Reversion (Bollinger Bands)
        data['BB_middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # 3. Momentum (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Volume
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # 5. ATR para stop loss dinámico
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        data['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        # Señales combinadas
        data['Signal'] = 0
        
        # Condiciones de COMPRA
        buy_conditions = (
            (data['Close'] > data['EMA_20']) &  # Tendencia alcista
            (data['EMA_20'] > data['EMA_50']) &  # Confirmación de tendencia
            (data['RSI'] < 70) &  # No sobrecomprado
            (data['Close'] < data['BB_upper']) &  # No en extremo superior
            (data['Volume_Ratio'] > 1.2)  # Volumen alto
        )
        
        # Condiciones de VENTA
        sell_conditions = (
            (data['Close'] < data['EMA_20']) |  # Ruptura de tendencia
            (data['RSI'] > 80) |  # Sobrecomprado extremo
            (data['Close'] > data['BB_upper'] * 1.01)  # Fuera de Bollinger
        )
        
        data.loc[buy_conditions, 'Signal'] = 1
        data.loc[sell_conditions, 'Signal'] = -1
        
        return data
    
    def backtest_with_risk_management(self, symbol):
        """Backtest con gestión de riesgo profesional"""
        print(f"\n💼 Analizando {symbol}...")
        
        data = self.advanced_strategy(symbol)
        position = 0
        entry_price = 0
        stop_loss = 0
        trades = []
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            atr = data['ATR'].iloc[i]
            
            # GESTIÓN DE POSICIÓN ABIERTA
            if position > 0:
                # Trailing stop loss
                new_stop = current_price - (2 * atr)
                stop_loss = max(stop_loss, new_stop)
                
                # Verificar stop loss
                if current_price <= stop_loss:
                    profit = (current_price - entry_price) / entry_price
                    trades.append({
                        'entry': entry_price,
                        'exit': current_price,
                        'profit': profit,
                        'type': 'Stop Loss'
                    })
                    position = 0
                    continue
            
            # SEÑALES DE ENTRADA/SALIDA
            if signal == 1 and position == 0:
                # Tamaño de posición basado en riesgo
                risk_amount = self.capital * self.risk_per_trade
                stop_distance = 2 * atr
                position_size = risk_amount / stop_distance
                
                position = min(position_size, self.capital * 0.25)  # Máximo 25% por posición
                entry_price = current_price
                stop_loss = current_price - stop_distance
                
            elif signal == -1 and position > 0:
                profit = (current_price - entry_price) / entry_price
                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'type': 'Signal'
                })
                position = 0
        
        # Calcular estadísticas
        if trades:
            profits = [t['profit'] for t in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            
            # Expectativa matemática
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Sharpe Ratio
            returns = pd.Series(profits)
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            return {
                'symbol': symbol,
                'trades': len(trades),
                'win_rate': win_rate,
                'expectancy': expectancy,
                'sharpe': sharpe,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_return': sum(profits)
            }
        
        return None
    
    def run_portfolio_analysis(self):
        """Análisis completo del portfolio"""
        print("\n" + "="*60)
        print("🚀 SISTEMA DE TRADING PROFESIONAL")
        print("="*60)
        print(f"Capital inicial: ${self.initial_capital:,}")
        print(f"Riesgo por operación: {self.risk_per_trade*100}%")
        
        # Encontrar las mejores acciones
        top_stocks = self.find_momentum_stocks()
        
        # Analizar cada acción
        results = []
        for symbol in top_stocks:
            result = self.backtest_with_risk_management(symbol)
            if result:
                results.append(result)
        
        # Ordenar por expectancia
        results.sort(key=lambda x: x['expectancy'], reverse=True)
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL ANÁLISIS")
        print("="*60)
        
        for r in results:
            print(f"\n{r['symbol']}:")
            print(f"  - Operaciones: {r['trades']}")
            print(f"  - Tasa de éxito: {r['win_rate']*100:.1f}%")
            print(f"  - Expectancia: {r['expectancy']*100:.2f}%")
            print(f"  - Sharpe Ratio: {r['sharpe']:.2f}")
            print(f"  - Retorno total: {r['total_return']*100:.1f}%")
        
        # Recomendaciones
        print("\n" + "="*60)
        print("💡 RECOMENDACIONES")
        print("="*60)
        
        if results:
            best = results[0]
            if best['expectancy'] > 0 and best['sharpe'] > 1:
                print(f"✅ MEJOR OPORTUNIDAD: {best['symbol']}")
                print(f"   - Expectancia positiva: {best['expectancy']*100:.1f}% por operación")
                print(f"   - Ratio riesgo/recompensa favorable")
                
                # Calcular Kelly
                kelly_pct = self.kelly_criterion(best['win_rate'], best['avg_win'], best['avg_loss'])
                print(f"   - Asignación óptima de capital: {kelly_pct*100:.1f}%")
                print(f"   - Monto sugerido: ${self.initial_capital * kelly_pct:,.0f}")
            else:
                print("⚠️ PRECAUCIÓN: No hay oportunidades claras en este momento")
                print("   - Considera esperar mejores condiciones de mercado")
                print("   - O usa paper trading para practicar")
        
        # Consejos finales
        print("\n" + "="*60)
        print("🎯 PLAN DE ACCIÓN PARA HACERTE RICO:")
        print("="*60)
        print("1. EDUCACIÓN: Dedica 1-2 horas diarias a aprender")
        print("2. PAPER TRADING: Practica 3-6 meses sin dinero real")
        print("3. EMPIEZA PEQUEÑO: Usa máximo $1,000 al principio")
        print("4. DIARIO DE TRADING: Registra TODAS tus operaciones")
        print("5. GESTIÓN DE RIESGO: NUNCA arriesgues más del 1% por operación")
        print("6. DIVERSIFICA: No todo en trading - invierte también en índices")
        print("7. PACIENCIA: Los ricos se hacen en años, no en días")
        
        print("\n⚡ SIGUIENTE PASO: Abre una cuenta demo en un broker")
        print("   Recomendados: Interactive Brokers, TD Ameritrade, Alpaca")

# Función para análisis rápido
def quick_opportunity_scan():
    """Escaneo rápido de oportunidades"""
    print("\n🔥 ESCANEANDO OPORTUNIDADES CALIENTES...\n")
    
    hot_stocks = {
        'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
        'AI Revolution': ['NVDA', 'AMD', 'PLTR', 'AI', 'MRVL'],
        'Crypto Stocks': ['COIN', 'MARA', 'RIOT', 'MSTR', 'SQ'],
        'High Growth': ['TSLA', 'SHOP', 'NET', 'DDOG', 'SNOW']
    }
    
    for category, symbols in hot_stocks.items():
        print(f"\n{category}:")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                if len(hist) > 0:
                    change_1w = (hist['Close'][-5:].iloc[-1] / hist['Close'][-5:].iloc[0] - 1) * 100
                    volume_avg = hist['Volume'][-5:].mean()
                    
                    # Señal caliente
                    if change_1w > 5 and volume_avg > hist['Volume'].mean():
                        print(f"  🔥 {symbol}: +{change_1w:.1f}% esta semana (CALIENTE!)")
                    elif change_1w > 0:
                        print(f"  ✅ {symbol}: +{change_1w:.1f}% esta semana")
                    else:
                        print(f"  ❌ {symbol}: {change_1w:.1f}% esta semana")
            except:
                continue

if __name__ == "__main__":
    # Sistema completo
    system = ProTradingSystem(initial_capital=10000)
    system.run_portfolio_analysis()
    
    # Escaneo rápido adicional
    print("\n" + "="*60)
    quick_opportunity_scan()