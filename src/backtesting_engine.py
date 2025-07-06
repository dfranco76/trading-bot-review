# src/backtesting_engine.py
"""
Sistema de backtesting para validar estrategias con datos históricos
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List
import json

class BacktestingEngine:
    def __init__(self, initial_capital=200):
        self.initial_capital = initial_capital
        self.results = []
        
    def backtest_strategy(self, symbol, strategy_func, start_date, end_date, params={}):
        """Ejecuta backtest de una estrategia"""
        # Descargar datos históricos
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) < 50:
            print(f"⚠️ Datos insuficientes para {symbol}")
            return None
        
        # Variables de estado
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        # Ejecutar estrategia día por día
        for i in range(20, len(data)):
            # Obtener slice de datos hasta el día actual
            historical_data = data.iloc[:i+1]
            
            # Generar señal
            signal = strategy_func(symbol, historical_data, params)
            
            if signal['action'] == 'BUY' and position == 0:
                # Comprar
                shares = (capital * 0.95) / signal['price']  # 95% del capital
                position = shares
                capital -= shares * signal['price']
                
                trades.append({
                    'date': data.index[i],
                    'action': 'BUY',
                    'price': signal['price'],
                    'shares': shares,
                    'reason': signal.get('reason', '')
                })
                
            elif signal['action'] == 'SELL' and position > 0:
                # Vender
                capital += position * signal['price']
                
                # Calcular P&L
                buy_price = trades[-1]['price'] if trades else signal['price']
                pnl = (signal['price'] - buy_price) * position
                pnl_pct = ((signal['price'] - buy_price) / buy_price) * 100
                
                trades.append({
                    'date': data.index[i],
                    'action': 'SELL',
                    'price': signal['price'],
                    'shares': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': signal.get('reason', '')
                })
                
                position = 0
            
            # Actualizar valor de cartera
            portfolio_value = capital + (position * data['Close'].iloc[i])
            equity_curve.append(portfolio_value)
        
        # Cerrar posición abierta al final
        if position > 0:
            final_price = data['Close'].iloc[-1]
            capital += position * final_price
            portfolio_value = capital
        
        # Calcular métricas
        metrics = self.calculate_metrics(trades, equity_curve, data)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'final_capital': portfolio_value
        }
    
    def calculate_metrics(self, trades, equity_curve, price_data):
        """Calcula métricas de performance"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Filtrar solo trades cerrados (con P&L)
        closed_trades = [t for t in trades if 'pnl' in t]
        
        if not closed_trades:
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Win rate
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(closed_trades)
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252) if equity_returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Total return
        total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
        
        # Estadísticas adicionales
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in closed_trades if t['pnl'] < 0]) if gross_loss > 0 else 0
        
        return {
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': max(closed_trades, key=lambda x: x['pnl'])['pnl'] if closed_trades else 0,
            'worst_trade': min(closed_trades, key=lambda x: x['pnl'])['pnl'] if closed_trades else 0
        }
    
    def backtest_multi_agent(self, symbols, agents, start_date, end_date):
        """Backtest del sistema multi-agente completo"""
        results = {}
        
        for symbol in symbols:
            print(f"\n🔍 Backtesting {symbol}...")
            
            # Simular votación de agentes
            def multi_agent_strategy(sym, data, params):
                votes = []
                
                # Simular voto de cada agente
                for agent in agents:
                    vote = agent.analyze_historical(sym, data)
                    votes.append(vote)
                
                # Sistema de consenso
                buy_votes = sum(1 for v in votes if v['action'] == 'BUY')
                sell_votes = sum(1 for v in votes if v['action'] == 'SELL')
                
                if buy_votes >= 3:
                    return {'action': 'BUY', 'price': data['Close'].iloc[-1]}
                elif sell_votes >= 3:
                    return {'action': 'SELL', 'price': data['Close'].iloc[-1]}
                else:
                    return {'action': 'HOLD', 'price': data['Close'].iloc[-1]}
            
            result = self.backtest_strategy(
                symbol,
                multi_agent_strategy,
                start_date,
                end_date
            )
            
            if result:
                results[symbol] = result
        
        return results
    
    def generate_report(self, results, save_path='backtest_report.html'):
        """Genera reporte HTML con resultados"""
        html = """
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #4CAF50; color: white; }
                .positive { color: green; }
                .negative { color: red; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>📊 Reporte de Backtesting</h1>
            <h2>Período: """ + f"{results[list(results.keys())[0]]['trades'][0]['date'].strftime('%Y-%m-%d')} - {datetime.now().strftime('%Y-%m-%d')}" + """</h2>
            
            <h3>Resumen de Performance</h3>
            <table>
                <tr>
                    <th>Símbolo</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Sharpe Ratio</th>
                    <th>Max DD</th>
                    <th>Return</th>
                </tr>
        """
        
        for symbol, data in results.items():
            metrics = data['metrics']
            return_class = 'positive' if metrics['total_return'] > 0 else 'negative'
            
            html += f"""
                <tr>
                    <td><b>{symbol}</b></td>
                    <td>{metrics['total_trades']}</td>
                    <td>{metrics['win_rate']*100:.1f}%</td>
                    <td>{metrics['profit_factor']:.2f}</td>
                    <td>{metrics['sharpe_ratio']:.2f}</td>
                    <td class="negative">{metrics['max_drawdown']:.1f}%</td>
                    <td class="{return_class}">{metrics['total_return']:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Estadísticas Detalladas</h3>
            <ul>
        """
        
        total_return = np.mean([r['metrics']['total_return'] for r in results.values()])
        avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results.values()])
        
        html += f"""
                <li>Return promedio: {total_return:.1f}%</li>
                <li>Sharpe promedio: {avg_sharpe:.2f}</li>
                <li>Símbolos analizados: {len(results)}</li>
            </ul>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"\n📄 Reporte guardado en: {save_path}")
    
    def plot_equity_curve(self, results):
        """Grafica curva de equity"""
        plt.figure(figsize=(12, 6))
        
        for symbol, data in results.items():
            plt.plot(data['equity_curve'], label=symbol)
        
        plt.title('Curva de Equity - Backtesting')
        plt.xlabel('Días')
        plt.ylabel('Valor de Cartera ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png')
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear engine
    backtest = BacktestingEngine(initial_capital=200)
    
    # Definir estrategia simple para test
    def simple_momentum_strategy(symbol, data, params):
        if len(data) < 20:
            return {'action': 'HOLD', 'price': data['Close'].iloc[-1]}
        
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if current_price > sma_20 * 1.02:
            return {'action': 'BUY', 'price': current_price}
        elif current_price < sma_20 * 0.98:
            return {'action': 'SELL', 'price': current_price}
        else:
            return {'action': 'HOLD', 'price': current_price}
    
    # Ejecutar backtest
    result = backtest.backtest_strategy(
        'NVDA',
        simple_momentum_strategy,
        datetime.now() - timedelta(days=180),
        datetime.now()
    )
    
    if result:
        print(f"\n📊 Resultados del Backtest:")
        print(f"Capital final: ${result['final_capital']:.2f}")
        print(f"Return: {result['metrics']['total_return']:.1f}%")
        print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['metrics']['max_drawdown']:.1f}%")