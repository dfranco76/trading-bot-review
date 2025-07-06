# automated_reporting_system.py
"""
Sistema completo de generación automatizada de reportes
Incluye análisis detallado, visualizaciones y alertas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import io
import base64
import time

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AutomatedReportingSystem:
    def __init__(self, supabase_client=None, notification_system=None):
        self.supabase = supabase_client
        self.notifications = notification_system
        self.report_data = {}
        
        # Plantillas HTML para reportes
        self.html_templates = {
            'daily': self._load_daily_template(),
            'weekly': self._load_weekly_template(),
            'alerts': self._load_alerts_template()
        }
        
        print("📊 Sistema de Reportes Automatizados iniciado")
    
    def _load_daily_template(self) -> str:
        """Plantilla HTML para reporte diario"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Reporte Diario de Trading - {date}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .positive {{
                    color: #27ae60;
                }}
                .negative {{
                    color: #e74c3c;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                }}
                .chart {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .alert {{
                    background-color: #f39c12;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .recommendation {{
                    background-color: #3498db;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Reporte Diario de Trading</h1>
                <p>{date} | Sistema Multi-Agente v5.0</p>
            </div>
            
            <div class="section">
                <h2>📈 Resumen del Día</h2>
                <div class="metrics">
                    {summary_metrics}
                </div>
            </div>
            
            <div class="section">
                <h2>💼 Trades Ejecutados</h2>
                {trades_table}
            </div>
            
            <div class="section">
                <h2>📊 Performance por Símbolo</h2>
                {symbol_performance}
                <div class="chart">
                    {performance_chart}
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 Análisis por Agente</h2>
                {agent_analysis}
            </div>
            
            <div class="section">
                <h2>⚠️ Alertas y Riesgos</h2>
                {alerts}
            </div>
            
            <div class="section">
                <h2>💡 Recomendaciones para Mañana</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>📈 Curva de Equity</h2>
                <div class="chart">
                    {equity_chart}
                </div>
            </div>
            
            <footer style="text-align: center; color: #7f8c8d; margin-top: 40px;">
                <p>Generado automáticamente por Trading Bot Multi-Agente</p>
                <p>© 2024 - Todos los derechos reservados</p>
            </footer>
        </body>
        </html>
        """
    
    def _load_weekly_template(self) -> str:
        """Plantilla HTML para reporte semanal"""
        return """
        <!-- Template similar pero con métricas semanales -->
        """
    
    def _load_alerts_template(self) -> str:
        """Plantilla HTML para alertas críticas"""
        return """
        <!-- Template para alertas urgentes -->
        """
    
    def generate_daily_report(self) -> Dict:
        """Genera reporte diario completo"""
        print("\n📊 Generando reporte diario...")
        
        # Recopilar datos
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades': self._get_daily_trades(),
            'performance': self._calculate_daily_performance(),
            'agent_stats': self._analyze_agent_performance(),
            'market_conditions': self._get_market_conditions(),
            'risk_metrics': self._calculate_risk_metrics(),
            'ml_performance': self._get_ml_performance()
        }
        
        # Generar secciones del reporte
        html_sections = {
            'summary_metrics': self._generate_summary_metrics(report_data),
            'trades_table': self._generate_trades_table(report_data['trades']),
            'symbol_performance': self._generate_symbol_performance(report_data),
            'agent_analysis': self._generate_agent_analysis(report_data['agent_stats']),
            'alerts': self._generate_alerts(report_data),
            'recommendations': self._generate_recommendations(report_data),
            'performance_chart': self._generate_performance_chart(report_data),
            'equity_chart': self._generate_equity_chart()
        }
        
        # Compilar HTML
        html_content = self.html_templates['daily'].format(
            date=report_data['date'],
            **html_sections
        )
        
        # Guardar reporte
        filename = f"daily_report_{report_data['date']}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Reporte diario guardado: {filename}")
        
        # Enviar por email si está configurado
        if self.notifications:
            self._send_email_report(filename, html_content, "Reporte Diario")
        
        return report_data
    
    def _get_daily_trades(self) -> List[Dict]:
        """Obtiene trades del día"""
        if not self.supabase:
            return []
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            response = self.supabase.table('trades')\
                .select("*")\
                .gte('created_at', today)\
                .execute()
            
            return response.data
        except:
            return []
    
    def _calculate_daily_performance(self) -> Dict:
        """Calcula métricas de performance del día"""
        trades = self._get_daily_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'average_win': 0,
                'average_loss': 0,
                'profit_factor': 0
            }
        
        # Calcular métricas
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _analyze_agent_performance(self) -> Dict:
        """Analiza performance por agente"""
        trades = self._get_daily_trades()
        
        agent_stats = {}
        agent_names = ['Momentum', 'Mean Reversion', 'Pattern', 'Volume', 'Sentiment']
        
        for agent in agent_names:
            agent_trades = [t for t in trades if agent in t.get('agent_name', '')]
            
            if agent_trades:
                wins = sum(1 for t in agent_trades if t.get('pnl', 0) > 0)
                total = len(agent_trades)
                avg_confidence = np.mean([t.get('agent_confidence', 0.5) for t in agent_trades])
                
                agent_stats[agent] = {
                    'trades': total,
                    'wins': wins,
                    'win_rate': wins / total if total > 0 else 0,
                    'avg_confidence': avg_confidence
                }
            else:
                agent_stats[agent] = {
                    'trades': 0,
                    'wins': 0,
                    'win_rate': 0,
                    'avg_confidence': 0
                }
        
        return agent_stats
    
    def _get_market_conditions(self) -> Dict:
        """Obtiene condiciones actuales del mercado"""
        # Implementar análisis de condiciones de mercado
        return {
            'volatility': 'normal',
            'trend': 'neutral',
            'volume': 'average'
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calcula métricas de riesgo"""
        # Implementar cálculo de métricas de riesgo
        return {
            'max_drawdown': 0,
            'var_95': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
    
    def _get_ml_performance(self) -> Dict:
        """Obtiene performance de modelos ML"""
        if not self.supabase:
            return {}
        
        try:
            # Obtener predicciones ML del día
            today = datetime.now().strftime('%Y-%m-%d')
            response = self.supabase.table('ml_features')\
                .select("ml_prediction, actual_best_action")\
                .gte('timestamp', today)\
                .not_.is_('actual_best_action', None)\
                .execute()
            
            if not response.data:
                return {'accuracy': 0, 'predictions': 0}
            
            correct = sum(1 for r in response.data 
                         if r['ml_prediction'] == r['actual_best_action'])
            total = len(response.data)
            
            return {
                'accuracy': correct / total if total > 0 else 0,
                'predictions': total,
                'correct': correct
            }
        except:
            return {'accuracy': 0, 'predictions': 0}
    
    def _generate_summary_metrics(self, data: Dict) -> str:
        """Genera HTML para métricas de resumen"""
        perf = data['performance']
        
        pnl_class = 'positive' if perf['total_pnl'] >= 0 else 'negative'
        
        html = f"""
        <div class="metric">
            <div class="metric-value {pnl_class}">${perf['total_pnl']:.2f}</div>
            <div class="metric-label">P&L Total</div>
        </div>
        <div class="metric">
            <div class="metric-value">{perf['total_trades']}</div>
            <div class="metric-label">Trades Totales</div>
        </div>
        <div class="metric">
            <div class="metric-value">{perf['win_rate']*100:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{perf['profit_factor']:.2f}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
        """
        
        return html
    
    def _generate_trades_table(self, trades: List[Dict]) -> str:
        """Genera tabla HTML de trades"""
        if not trades:
            return "<p>No hay trades ejecutados hoy</p>"
        
        html = """
        <table>
            <tr>
                <th>Hora</th>
                <th>Símbolo</th>
                <th>Acción</th>
                <th>Precio</th>
                <th>Cantidad</th>
                <th>P&L</th>
                <th>Consenso</th>
                <th>Confianza</th>
            </tr>
        """
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            pnl_class = 'positive' if pnl >= 0 else 'negative'
            
            html += f"""
            <tr>
                <td>{trade.get('created_at', '')[:10]}</td>
                <td><strong>{trade.get('symbol', '')}</strong></td>
                <td>{trade.get('action', '')}</td>
                <td>${trade.get('price', 0):.2f}</td>
                <td>{trade.get('quantity', 0):.2f}</td>
                <td class="{pnl_class}">${pnl:.2f}</td>
                <td>{trade.get('consensus_type', '')}</td>
                <td>{trade.get('agent_confidence', 0)*100:.0f}%</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_symbol_performance(self, data: Dict) -> str:
        """Genera análisis de performance por símbolo"""
        trades = data['trades']
        
        if not trades:
            return "<p>No hay datos de símbolos</p>"
        
        # Agrupar por símbolo
        symbol_stats = {}
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'trades': 0,
                    'pnl': 0,
                    'volume': 0
                }
            
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade.get('pnl', 0)
            symbol_stats[symbol]['volume'] += trade.get('quantity', 0) * trade.get('price', 0)
        
        # Generar HTML
        html = "<table><tr><th>Símbolo</th><th>Trades</th><th>P&L</th><th>Volumen</th></tr>"
        
        for symbol, stats in sorted(symbol_stats.items(), 
                                   key=lambda x: x[1]['pnl'], 
                                   reverse=True):
            pnl_class = 'positive' if stats['pnl'] >= 0 else 'negative'
            html += f"""
            <tr>
                <td><strong>{symbol}</strong></td>
                <td>{stats['trades']}</td>
                <td class="{pnl_class}">${stats['pnl']:.2f}</td>
                <td>${stats['volume']:.2f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_agent_analysis(self, agent_stats: Dict) -> str:
        """Genera análisis de agentes"""
        html = "<table><tr><th>Agente</th><th>Trades</th><th>Ganados</th><th>Win Rate</th><th>Confianza Promedio</th></tr>"
        
        for agent, stats in agent_stats.items():
            html += f"""
            <tr>
                <td><strong>{agent}</strong></td>
                <td>{stats['trades']}</td>
                <td>{stats['wins']}</td>
                <td>{stats['win_rate']*100:.1f}%</td>
                <td>{stats['avg_confidence']*100:.0f}%</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_alerts(self, data: Dict) -> str:
        """Genera alertas basadas en el análisis"""
        alerts = []
        
        # Verificar performance
        if data['performance']['total_pnl'] < -50:
            alerts.append("Pérdidas significativas detectadas - Revisar estrategia")
        
        if data['performance']['win_rate'] < 0.3:
            alerts.append("Win rate muy bajo - Considerar pausar trading")
        
        # Verificar riesgo
        if data['risk_metrics'].get('max_drawdown', 0) < -10:
            alerts.append("Drawdown elevado - Reducir tamaño de posiciones")
        
        if not alerts:
            return '<p style="color: green;">✅ No hay alertas críticas</p>'
        
        html = ""
        for alert in alerts:
            html += f'<div class="alert">⚠️ {alert}</div>'
        
        return html
    
    def _generate_recommendations(self, data: Dict) -> str:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Basado en performance
        if data['performance']['win_rate'] > 0.6:
            recommendations.append("Performance sólido - Considerar aumentar tamaño de posiciones gradualmente")
        
        # Basado en agentes
        best_agent = max(data['agent_stats'].items(), 
                        key=lambda x: x[1]['win_rate'] if x[1]['trades'] > 0 else 0)
        if best_agent[1]['win_rate'] > 0.7 and best_agent[1]['trades'] > 3:
            recommendations.append(f"Agente {best_agent[0]} muestra excelente performance - Aumentar su peso")
        
        # Basado en ML
        ml_perf = data.get('ml_performance', {})
        if ml_perf.get('accuracy', 0) > 0.65:
            recommendations.append("Modelos ML muestran buena precisión - Considerar aumentar su influencia")
        
        # Basado en mercado
        if data['market_conditions']['volatility'] == 'high':
            recommendations.append("Alta volatilidad detectada - Reducir tamaños y ser más selectivo")
        
        if not recommendations:
            recommendations.append("Mantener estrategia actual y monitorear de cerca")
        
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">💡 {rec}</div>'
        
        return html
    
    def _generate_performance_chart(self, data: Dict) -> str:
        """Genera gráfico de performance"""
        try:
            trades = data['trades']
            if not trades:
                return "<p>No hay datos para graficar</p>"
            
            # Crear gráfico de P&L acumulado
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Ordenar trades por tiempo
            trades_sorted = sorted(trades, key=lambda x: x.get('created_at', ''))
            
            # Calcular P&L acumulado
            timestamps = []
            cumulative_pnl = []
            current_pnl = 0
            
            for trade in trades_sorted:
                current_pnl += trade.get('pnl', 0)
                timestamps.append(trade.get('created_at', ''))
                cumulative_pnl.append(current_pnl)
            
            # Graficar
            ax.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2)
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            ax.set_title('P&L Acumulado del Día')
            ax.set_xlabel('Trades')
            ax.set_ylabel('P&L ($)')
            ax.grid(True, alpha=0.3)
            
            # Guardar como imagen base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{image_base64}" width="100%">'
            
        except Exception as e:
            print(f"Error generando gráfico: {e}")
            return "<p>Error generando gráfico</p>"
    
    def _generate_equity_chart(self) -> str:
        """Genera gráfico de curva de equity"""
        # Similar a _generate_performance_chart pero con datos históricos
        return "<p>Gráfico de equity en desarrollo</p>"
    
    def _send_email_report(self, filename: str, html_content: str, subject: str):
        """Envía reporte por email"""
        try:
            # Usar el sistema de notificaciones si está disponible
            if hasattr(self.notifications, 'send_email'):
                self.notifications.send_email(subject, html_content)
                print(f"✅ Reporte enviado por email: {subject}")
        except Exception as e:
            print(f"❌ Error enviando email: {e}")
    
    def generate_weekly_summary(self) -> Dict:
        """Genera resumen semanal"""
        print("\n📊 Generando resumen semanal...")
        
        # Obtener datos de la semana
        week_start = datetime.now() - timedelta(days=7)
        
        if self.supabase:
            response = self.supabase.table('trades')\
                .select("*")\
                .gte('created_at', week_start.isoformat())\
                .execute()
            
            weekly_trades = response.data
        else:
            weekly_trades = []
        
        # Calcular métricas semanales
        total_pnl = sum(t.get('pnl', 0) for t in weekly_trades)
        total_trades = len(weekly_trades)
        winning_trades = sum(1 for t in weekly_trades if t.get('pnl', 0) > 0)
        
        summary = {
            'period': f"{week_start.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'best_day': self._find_best_day(weekly_trades),
            'worst_day': self._find_worst_day(weekly_trades),
            'most_traded': self._find_most_traded_symbol(weekly_trades)
        }
        
        print("✅ Resumen semanal generado")
        return summary
    
    def _find_best_day(self, trades: List[Dict]) -> Dict:
        """Encuentra el mejor día de la semana"""
        daily_pnl = {}
        
        for trade in trades:
            date = trade.get('created_at', '')[:10]
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += trade.get('pnl', 0)
        
        if not daily_pnl:
            return {'date': 'N/A', 'pnl': 0}
        
        best_day = max(daily_pnl.items(), key=lambda x: x[1])
        return {'date': best_day[0], 'pnl': best_day[1]}
    
    def _find_worst_day(self, trades: List[Dict]) -> Dict:
        """Encuentra el peor día de la semana"""
        daily_pnl = {}
        
        for trade in trades:
            date = trade.get('created_at', '')[:10]
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += trade.get('pnl', 0)
        
        if not daily_pnl:
            return {'date': 'N/A', 'pnl': 0}
        
        worst_day = min(daily_pnl.items(), key=lambda x: x[1])
        return {'date': worst_day[0], 'pnl': worst_day[1]}
    
    def _find_most_traded_symbol(self, trades: List[Dict]) -> str:
        """Encuentra el símbolo más tradeado"""
        symbol_counts = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        if not symbol_counts:
            return 'N/A'
        
        return max(symbol_counts.items(), key=lambda x: x[1])[0]
    
    def schedule_reports(self):
        """Programa generación automática de reportes"""
        import schedule
        
        # Reporte diario a las 22:30
        schedule.every().day.at("22:30").do(self.generate_daily_report)
        
        # Resumen semanal los domingos a las 23:00
        schedule.every().sunday.at("23:00").do(self.generate_weekly_summary)
        
        print("📅 Reportes programados:")
        print("   • Diario: 22:30")
        print("   • Semanal: Domingos 23:00")
        
        # Loop de ejecución
        while True:
            schedule.run_pending()
            time.sleep(60)