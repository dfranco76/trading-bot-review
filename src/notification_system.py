# src/notification_system.py
"""
Sistema de notificaciones para alertas importantes del bot
"""
import os
from datetime import datetime
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

class NotificationSystem:
    def __init__(self):
        # Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Discord
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Email
        self.email_smtp = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.email_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_pass = os.getenv('EMAIL_PASS')
        self.email_to = os.getenv('EMAIL_TO')
        
        # Configuración
        self.channels = {
            'telegram': bool(self.telegram_token),
            'discord': bool(self.discord_webhook),
            'email': bool(self.email_user and self.email_pass)
        }
        
        print(f"📱 Sistema de notificaciones iniciado")
        print(f"   Canales activos: {[k for k,v in self.channels.items() if v]}")
    
    def send_telegram(self, message):
        """Envía notificación por Telegram"""
        if not self.channels['telegram']:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            requests.post(url, data=data)
        except Exception as e:
            print(f"Error enviando Telegram: {e}")
    
    def send_discord(self, title, description, color=0x00ff00):
        """Envía notificación por Discord"""
        if not self.channels['discord']:
            return
            
        try:
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "Trading Bot v5.0"}
            }
            
            data = {"embeds": [embed]}
            requests.post(self.discord_webhook, json=data)
        except Exception as e:
            print(f"Error enviando Discord: {e}")
    
    def send_email(self, subject, body):
        """Envía notificación por email"""
        if not self.channels['email']:
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.email_to
            msg['Subject'] = f"[Trading Bot] {subject}"
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.email_smtp, self.email_port)
            server.starttls()
            server.login(self.email_user, self.email_pass)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Error enviando email: {e}")
    
    def notify_trade(self, trade_info):
        """Notifica sobre un nuevo trade"""
        emoji = "🟢" if trade_info['action'] == 'BUY' else "🔴"
        
        # Telegram
        telegram_msg = f"""
{emoji} <b>NUEVO TRADE</b>

📊 {trade_info['symbol']} - {trade_info['action']}
💰 Precio: ${trade_info['price']:.2f}
📈 Cantidad: {trade_info['quantity']:.2f}
💪 Confianza: {trade_info['confidence']*100:.0f}%
📝 Tipo: {trade_info['consensus_type']}
        """
        self.send_telegram(telegram_msg)
        
        # Discord
        color = 0x00ff00 if trade_info['action'] == 'BUY' else 0xff0000
        self.send_discord(
            f"{emoji} {trade_info['action']} {trade_info['symbol']}",
            f"Precio: ${trade_info['price']:.2f}\nConfianza: {trade_info['confidence']*100:.0f}%",
            color
        )
        
        # Email (solo trades importantes)
        if trade_info['confidence'] > 0.8 or trade_info['quantity'] * trade_info['price'] > 100:
            self.send_email(
                f"Trade Ejecutado: {trade_info['symbol']}",
                f"<h2>Nuevo Trade</h2><p>Detalles: {trade_info}</p>"
            )
    
    def notify_daily_summary(self, summary):
        """Envía resumen diario"""
        emoji = "📈" if summary['pnl'] >= 0 else "📉"
        
        message = f"""
{emoji} <b>RESUMEN DIARIO</b>

💰 P&L: ${summary['pnl']:.2f} ({summary['pnl_pct']:+.1f}%)
📊 Trades: {summary['total_trades']}
✅ Ganados: {summary['wins']} ({summary['win_rate']*100:.0f}%)
❌ Perdidos: {summary['losses']}

🏆 Mejor: {summary['best_trade']}
💔 Peor: {summary['worst_trade']}
        """
        
        self.send_telegram(message)
        self.send_discord("📊 Resumen Diario", message.replace('<b>', '**').replace('</b>', '**'))
        self.send_email("Resumen Diario de Trading", message.replace('\n', '<br>'))
    
    def notify_alert(self, alert_type, message, critical=False):
        """Envía alertas del sistema"""
        emoji_map = {
            'risk': '⚠️',
            'error': '❌',
            'warning': '⚡',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        emoji = emoji_map.get(alert_type, '📢')
        
        if critical:
            # Enviar por todos los canales si es crítico
            self.send_telegram(f"{emoji} <b>ALERTA CRÍTICA</b>\n\n{message}")
            self.send_discord(f"{emoji} Alerta Crítica", message, 0xff0000)
            self.send_email("ALERTA CRÍTICA", message)
        else:
            # Solo Telegram/Discord para alertas normales
            self.send_telegram(f"{emoji} {message}")
            self.send_discord(f"{emoji} Alerta", message)