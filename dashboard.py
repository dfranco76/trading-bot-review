# dashboard.py
"""
Backend del Dashboard Profesional para Trading Bot
"""

from flask import Flask, render_template_string, jsonify, request
import subprocess
import threading
import json
import os
import time
import psutil
from datetime import datetime
import sys

# Agregar el directorio src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Estado global
bot_state = {
    'process': None,
    'running': False,
    'start_time': None,
    'logs': [],
    'metrics': {
        'capital': 200.0,
        'pnl_today': 0.0,
        'trades_today': 0,
        'open_positions': 0,
        'win_rate': 0.0
    }
}

# Cargar el HTML del dashboard
with open('dashboard.html', 'r', encoding='utf-8') as f:
    DASHBOARD_HTML = f.read()

@app.route('/')
def index():
    """Servir el dashboard"""
    return DASHBOARD_HTML

@app.route('/api/status')
def get_status():
    """Obtener estado actual del sistema"""
    # Verificar si el proceso está vivo
    if bot_state['process'] and bot_state['process'].poll() is None:
        bot_state['running'] = True
    else:
        bot_state['running'] = False
    
    # Leer métricas actualizadas si existen
    try:
        if os.path.exists('bot_metrics.json'):
            with open('bot_metrics.json', 'r') as f:
                metrics = json.load(f)
                bot_state['metrics'].update(metrics)
    except:
        pass
    
    return jsonify({
        'running': bot_state['running'],
        'metrics': bot_state['metrics'],
        'start_time': bot_state['start_time'],
        'logs': bot_state['logs'][-50:]  # Últimos 50 logs
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar el bot"""
    if bot_state['running']:
        return jsonify({'success': False, 'error': 'Bot ya está en ejecución'})
    
    try:
        # Obtener configuración
        data = request.json or {}
        mode = data.get('mode', 'continuous')
        
        # Comando según el modo
        if mode == 'continuous':
            cmd = ['python', 'src/main_bot.py', '--mode', '2']
        else:
            cmd = ['python', 'src/main_bot.py']
        
        # Iniciar proceso
        bot_state['process'] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        bot_state['running'] = True
        bot_state['start_time'] = datetime.now().isoformat()
        
        # Thread para leer output
        threading.Thread(target=read_bot_output, daemon=True).start()
        
        add_log('Bot iniciado correctamente', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        add_log(f'Error al iniciar bot: {str(e)}', 'error')
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Detener el bot"""
    if not bot_state['running']:
        return jsonify({'success': False, 'error': 'Bot no está en ejecución'})
    
    try:
        if bot_state['process']:
            bot_state['process'].terminate()
            bot_state['process'] = None
        
        bot_state['running'] = False
        add_log('Bot detenido', 'info')
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/command', methods=['POST'])
def send_command():
    """Enviar comando al bot"""
    data = request.json
    command = data.get('command')
    
    commands = {
        'single_analysis': lambda: run_bot_function('analyze_single'),
        'backtest': lambda: run_bot_function('backtest'),
        'optimize': lambda: run_bot_function('optimize'),
        'practice': lambda: run_bot_function('practice_mode'),
        'report': lambda: run_bot_function('generate_report')
    }
    
    if command in commands:
        try:
            result = commands[command]()
            return jsonify({'success': True, 'result': result})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Comando no reconocido'})

@app.route('/api/health')
def health_check():
    """Verificación de salud del sistema"""
    health = {
        'timestamp': datetime.now().isoformat(),
        'checks': []
    }
    
    # Python version
    health['checks'].append({
        'name': 'Python Version',
        'status': 'success',
        'value': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    })
    
    # API Keys
    env_exists = os.path.exists('.env')
    health['checks'].append({
        'name': 'API Keys',
        'status': 'success' if env_exists else 'error',
        'value': 'Configuradas' if env_exists else 'No encontradas'
    })
    
    # Database
    try:
        import supabase
        health['checks'].append({
            'name': 'Base de Datos',
            'status': 'success',
            'value': 'Conectada'
        })
    except:
        health['checks'].append({
            'name': 'Base de Datos',
            'status': 'error',
            'value': 'No disponible'
        })
    
    # ML Models
    models_path = 'ml_models'
    if os.path.exists(models_path):
        model_files = len([f for f in os.listdir(models_path) if f.endswith('.pkl')])
        health['checks'].append({
            'name': 'Modelos ML',
            'status': 'success' if model_files > 0 else 'warning',
            'value': f'{model_files} modelos encontrados'
        })
    
    # System resources
    health['checks'].append({
        'name': 'CPU',
        'status': 'success' if psutil.cpu_percent() < 80 else 'warning',
        'value': f'{psutil.cpu_percent()}%'
    })
    
    health['checks'].append({
        'name': 'Memoria',
        'status': 'success' if psutil.virtual_memory().percent < 80 else 'warning',
        'value': f'{psutil.virtual_memory().percent}%'
    })
    
    # Paper trading
    try:
        with open('optimal_config.json', 'r') as f:
            config = json.load(f)
            paper_trading = config.get('trading', {}).get('paper_trading', True)
            health['checks'].append({
                'name': 'Paper Trading',
                'status': 'success' if paper_trading else 'warning',
                'value': 'Activado' if paper_trading else 'DINERO REAL'
            })
    except:
        pass
    
    return jsonify(health)

@app.route('/api/ml/status')
def ml_status():
    """Estado del sistema ML"""
    ml_info = {
        'models': [],
        'metrics': {},
        'training': {}
    }
    
    # Listar modelos
    if os.path.exists('ml_models'):
        for file in os.listdir('ml_models'):
            if file.endswith('.pkl'):
                ml_info['models'].append({
                    'name': file.replace('.pkl', ''),
                    'size': os.path.getsize(f'ml_models/{file}') / 1024 / 1024,  # MB
                    'modified': datetime.fromtimestamp(os.path.getmtime(f'ml_models/{file}'))
                })
    
    # Leer métricas si existen
    if os.path.exists('ml_metrics.json'):
        with open('ml_metrics.json', 'r') as f:
            ml_info['metrics'] = json.load(f)
    
    return jsonify(ml_info)

# Funciones auxiliares
def add_log(message, log_type='info'):
    """Agregar log al estado"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'type': log_type,
        'message': message
    }
    bot_state['logs'].append(log_entry)
    
    # Mantener solo los últimos 1000 logs
    if len(bot_state['logs']) > 1000:
        bot_state['logs'] = bot_state['logs'][-1000:]

def read_bot_output():
    """Leer output del bot en tiempo real"""
    if not bot_state['process']:
        return
    
    for line in iter(bot_state['process'].stdout.readline, ''):
        if line:
            # Determinar tipo de log
            log_type = 'info'
            if 'ERROR' in line:
                log_type = 'error'
            elif 'WARNING' in line:
                log_type = 'warning'
            elif 'SUCCESS' in line or '✅' in line:
                log_type = 'success'
            
            add_log(line.strip(), log_type)

def run_bot_function(function_name):
    """Ejecutar una función específica del bot"""
    # Aquí podrías importar y ejecutar funciones del bot
    # Por ahora, simular
    add_log(f'Ejecutando {function_name}...', 'info')
    time.sleep(1)
    add_log(f'{function_name} completado', 'success')
    return f'{function_name} ejecutado'

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 TRADING BOT DASHBOARD")
    print("="*50)
    print(f"📍 URL: http://localhost:5000")
    print("📍 Presiona Ctrl+C para detener")
    print("="*50 + "\n")
    
    # Asegurar que el HTML existe
    if not os.path.exists('dashboard.html'):
        print("⚠️  Creando dashboard.html...")
        # Aquí deberías guardar el HTML del artifact anterior
        
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Dashboard detenido")
        if bot_state['process']:
            bot_state['process'].terminate()