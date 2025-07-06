# paper_trading_reporter.py - Sistema automático de reportes para paper trading
import json
import csv
from datetime import datetime
import yfinance as yf
import pandas as pd
from tabulate import tabulate
import os

class PaperTradingReporter:
    def __init__(self):
        self.trades_file = f"paper_trades_{datetime.now().strftime('%Y%m%d')}.json"
        self.csv_file = f"paper_trades_{datetime.now().strftime('%Y%m%d')}.csv"
        self.trades = self.load_trades()
        
    def load_trades(self):
        """Carga trades del día si existen"""
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_trades(self):
        """Guarda trades a archivo"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def registrar_trade(self, trade_info):
        """Registra un nuevo paper trade"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'hora': datetime.now().strftime('%H:%M:%S'),
            'symbol': trade_info['symbol'],
            'action': trade_info['decision'],
            'precio_entrada': trade_info['price'],
            'cantidad': trade_info.get('quantity', 0),
            'valor': trade_info.get('quantity', 0) * trade_info['price'],
            'confianza': trade_info['confidence'],
            'tipo_consenso': trade_info.get('tipo', ''),
            'razon_principal': self.get_razon_principal(trade_info),
            'precio_actual': trade_info['price'],  # Se actualizará después
            'pnl': 0,  # Se calculará después
            'pnl_pct': 0,  # Se calculará después
            'estado': 'ABIERTO'
        }
        
        self.trades.append(trade)
        self.save_trades()
        
        print(f"\n📝 TRADE REGISTRADO:")
        print(f"   {trade['hora']} - {trade['symbol']} {trade['action']} @ ${trade['precio_entrada']:.2f}")
        
        return trade
    
    def get_razon_principal(self, trade_info):
        """Extrae la razón principal del trade"""
        if 'votos' in trade_info:
            # Buscar el agente con mayor confianza que votó por la acción
            votos_accion = [v for v in trade_info['votos'] if v['action'] == trade_info['decision']]
            if votos_accion:
                mejor_voto = max(votos_accion, key=lambda x: x['confidence'])
                return f"{mejor_voto['agent']}: {mejor_voto['reason']}"
        return trade_info.get('tipo', 'Sin razón especificada')
    
    def actualizar_precios(self):
        """Actualiza precios actuales y calcula P&L"""
        print("\n📊 Actualizando precios...")
        
        for trade in self.trades:
            if trade['estado'] == 'ABIERTO':
                try:
                    # Obtener precio actual
                    stock = yf.Ticker(trade['symbol'])
                    data = stock.history(period="1d")
                    if len(data) > 0:
                        precio_actual = data['Close'].iloc[-1]
                        trade['precio_actual'] = precio_actual
                        
                        # Calcular P&L
                        if trade['action'] == 'BUY':
                            trade['pnl'] = (precio_actual - trade['precio_entrada']) * trade['cantidad']
                            trade['pnl_pct'] = ((precio_actual - trade['precio_entrada']) / trade['precio_entrada']) * 100
                        else:  # SELL
                            trade['pnl'] = (trade['precio_entrada'] - precio_actual) * trade['cantidad']
                            trade['pnl_pct'] = ((trade['precio_entrada'] - precio_actual) / trade['precio_entrada']) * 100
                        
                        print(f"   {trade['symbol']}: ${precio_actual:.2f} (P&L: {trade['pnl_pct']:+.2f}%)")
                except:
                    print(f"   ❌ Error actualizando {trade['symbol']}")
        
        self.save_trades()
    
    def generar_reporte_tiempo_real(self):
        """Genera reporte en tiempo real"""
        self.actualizar_precios()
        
        print("\n" + "="*80)
        print(f"📊 REPORTE PAPER TRADING - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        if not self.trades:
            print("\n📭 No hay trades registrados todavía")
            return
        
        # Preparar datos para tabla
        tabla_data = []
        total_pnl = 0
        
        for trade in self.trades:
            emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
            pnl_emoji = "📈" if trade['pnl'] >= 0 else "📉"
            
            tabla_data.append([
                trade['hora'],
                f"{emoji} {trade['symbol']}",
                trade['action'],
                f"${trade['precio_entrada']:.2f}",
                f"${trade['precio_actual']:.2f}",
                f"{pnl_emoji} {trade['pnl_pct']:+.2f}%",
                f"${trade['pnl']:+.2f}"
            ])
            
            total_pnl += trade['pnl']
        
        # Mostrar tabla
        headers = ["Hora", "Símbolo", "Tipo", "Entrada", "Actual", "P&L %", "P&L $"]
        print(tabulate(tabla_data, headers=headers, tablefmt="simple"))
        
        # Resumen
        trades_ganadores = sum(1 for t in self.trades if t['pnl'] > 0)
        trades_perdedores = sum(1 for t in self.trades if t['pnl'] < 0)
        win_rate = (trades_ganadores / len(self.trades) * 100) if self.trades else 0
        
        print(f"\n📈 RESUMEN:")
        print(f"  • Total trades: {len(self.trades)}")
        print(f"  • Ganadores: {trades_ganadores} | Perdedores: {trades_perdedores}")
        print(f"  • Win Rate: {win_rate:.0f}%")
        print(f"  • P&L Total: ${total_pnl:+.2f}")
        print(f"  • P&L Promedio: ${total_pnl/len(self.trades):+.2f}" if self.trades else "")
        
        # Mejor y peor trade
        if self.trades:
            mejor_trade = max(self.trades, key=lambda x: x['pnl_pct'])
            peor_trade = min(self.trades, key=lambda x: x['pnl_pct'])
            
            print(f"\n🏆 Mejor trade: {mejor_trade['symbol']} {mejor_trade['pnl_pct']:+.2f}%")
            print(f"💔 Peor trade: {peor_trade['symbol']} {peor_trade['pnl_pct']:+.2f}%")
    
    def generar_reporte_final(self):
        """Genera reporte final del día con análisis detallado"""
        self.actualizar_precios()
        
        print("\n" + "="*80)
        print(f"📊 REPORTE FINAL PAPER TRADING - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*80)
        
        if not self.trades:
            print("\n📭 No hubo trades hoy")
            return
        
        # Exportar a CSV
        self.exportar_csv()
        
        # Análisis por símbolo
        print("\n📈 ANÁLISIS POR SÍMBOLO:")
        simbolos = {}
        for trade in self.trades:
            if trade['symbol'] not in simbolos:
                simbolos[trade['symbol']] = {'trades': 0, 'pnl': 0, 'pnl_pct': []}
            
            simbolos[trade['symbol']]['trades'] += 1
            simbolos[trade['symbol']]['pnl'] += trade['pnl']
            simbolos[trade['symbol']]['pnl_pct'].append(trade['pnl_pct'])
        
        for symbol, data in sorted(simbolos.items(), key=lambda x: x[1]['pnl'], reverse=True):
            avg_pnl_pct = sum(data['pnl_pct']) / len(data['pnl_pct'])
            print(f"  • {symbol}: {data['trades']} trades, P&L: ${data['pnl']:+.2f} ({avg_pnl_pct:+.2f}% avg)")
        
        # Análisis por hora
        print("\n⏰ ANÁLISIS POR HORA:")
        horas = {}
        for trade in self.trades:
            hora = trade['hora'].split(':')[0]
            if hora not in horas:
                horas[hora] = {'trades': 0, 'pnl': 0}
            
            horas[hora]['trades'] += 1
            horas[hora]['pnl'] += trade['pnl']
        
        for hora in sorted(horas.keys()):
            print(f"  • {hora}:00 - {horas[hora]['trades']} trades, P&L: ${horas[hora]['pnl']:+.2f}")
        
        # Análisis de tipos de consenso
        print("\n🤝 ANÁLISIS POR TIPO DE CONSENSO:")
        tipos = {}
        for trade in self.trades:
            tipo = trade['tipo_consenso']
            if tipo not in tipos:
                tipos[tipo] = {'trades': 0, 'pnl': 0, 'win_rate': []}
            
            tipos[tipo]['trades'] += 1
            tipos[tipo]['pnl'] += trade['pnl']
            tipos[tipo]['win_rate'].append(1 if trade['pnl'] > 0 else 0)
        
        for tipo, data in sorted(tipos.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr = sum(data['win_rate']) / len(data['win_rate']) * 100 if data['win_rate'] else 0
            print(f"  • {tipo}: {data['trades']} trades, P&L: ${data['pnl']:+.2f}, WR: {wr:.0f}%")
        
        # Conclusiones y recomendaciones
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades) * 100
        
        print("\n💡 CONCLUSIONES:")
        if total_pnl > 0:
            print(f"  ✅ Día positivo: ${total_pnl:+.2f}")
        else:
            print(f"  ❌ Día negativo: ${total_pnl:+.2f}")
        
        if win_rate > 60:
            print(f"  ✅ Buen win rate: {win_rate:.0f}%")
        elif win_rate < 40:
            print(f"  ⚠️ Win rate bajo: {win_rate:.0f}% - Revisar estrategia")
        
        # Guardar resumen
        self.guardar_resumen_diario(total_pnl, win_rate)
        
        print(f"\n📁 Datos guardados en:")
        print(f"  • JSON: {self.trades_file}")
        print(f"  • CSV: {self.csv_file}")
    
    def exportar_csv(self):
        """Exporta trades a CSV para análisis en Excel"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Hora', 'Símbolo', 'Acción', 'Precio Entrada', 'Precio Actual', 
                           'P&L $', 'P&L %', 'Confianza', 'Tipo Consenso', 'Razón'])
            
            for trade in self.trades:
                writer.writerow([
                    trade['hora'],
                    trade['symbol'],
                    trade['action'],
                    trade['precio_entrada'],
                    trade['precio_actual'],
                    round(trade['pnl'], 2),
                    round(trade['pnl_pct'], 2),
                    round(trade['confianza'] * 100, 0),
                    trade['tipo_consenso'],
                    trade['razon_principal']
                ])
    
    def guardar_resumen_diario(self, total_pnl, win_rate):
        """Guarda resumen diario para histórico"""
        resumen_file = "paper_trading_historico.json"
        
        # Cargar histórico
        if os.path.exists(resumen_file):
            with open(resumen_file, 'r') as f:
                historico = json.load(f)
        else:
            historico = []
        
        # Añadir día actual
        resumen = {
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'trades': len(self.trades),
            'pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 1),
            'mejor_trade': max((t for t in self.trades), key=lambda x: x['pnl_pct'])['symbol'] if self.trades else '',
            'peor_trade': min((t for t in self.trades), key=lambda x: x['pnl_pct'])['symbol'] if self.trades else ''
        }
        
        historico.append(resumen)
        
        # Guardar
        with open(resumen_file, 'w') as f:
            json.dump(historico, f, indent=2)


# Instancia global para usar desde safe_trading.py
reporter = PaperTradingReporter()


def generar_reporte_rapido():
    """Función helper para generar reporte rápido"""
    reporter.generar_reporte_tiempo_real()


def generar_reporte_final():
    """Función helper para generar reporte final"""
    reporter.generar_reporte_final()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "reporte":
            # Generar reporte en tiempo real
            reporter = PaperTradingReporter()
            reporter.generar_reporte_tiempo_real()
        elif sys.argv[1] == "final":
            # Generar reporte final
            reporter = PaperTradingReporter()
            reporter.generar_reporte_final()
    else:
        print("\n📊 PAPER TRADING REPORTER")
        print("="*50)
        print("\nUso:")
        print("  python paper_trading_reporter.py reporte  # Reporte tiempo real")
        print("  python paper_trading_reporter.py final    # Reporte final del día")