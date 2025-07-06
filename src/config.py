# src/config.py
from dotenv import load_dotenv
import os
import json

# Cargar variables de entorno
load_dotenv()

# ===========================
# CONFIGURACIÓN BASE
# ===========================

# Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Alpaca Trading
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Notificaciones (nuevas)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
EMAIL_TO = os.getenv('EMAIL_TO')
EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))

# ===========================
# SÍMBOLOS DE TRADING
# ===========================

# Tech giants con alta liquidez
TECH_GIANTS = ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META']

# High volatility tech
HIGH_VOLATILITY = ['PLTR', 'SOFI', 'NET', 'COIN', 'ROKU', 'MARA', 'SQ', 'RIOT']

# Growth stocks
GROWTH_STOCKS = ['DDOG', 'SNOW', 'CRWD', 'ZS', 'SHOP', 'SE', 'MELI']

# Fintech & Crypto related
FINTECH_CRYPTO = ['PYPL', 'V', 'MA', 'HOOD', 'AFRM', 'UPST']

# Meme stocks (usar con extrema precaución)
MEME_STOCKS = ['GME', 'AMC']  # Alta volatilidad, alto riesgo

# Selección principal para trading
SYMBOLS = [
    # Tech establecidas con buena liquidez
    'NVDA', 'AMD', 'TSLA', 
    
    # High growth con volatilidad
    'PLTR', 'SOFI', 'NET',
    
    # Crypto-related
    'COIN', 'MARA', 
    
    # Fintech innovadoras
    'SQ', 'HOOD',
    
    # Cloud computing
    'DDOG', 'SNOW'
]

# Lista extendida para análisis más amplio
SYMBOLS_EXTENDED = TECH_GIANTS + HIGH_VOLATILITY[:5] + GROWTH_STOCKS[:3]

# Lista conservadora (menos volátil)
SYMBOLS_CONSERVATIVE = ['AAPL', 'MSFT', 'GOOGL', 'V', 'MA']

# ===========================
# CONFIGURACIÓN OPTIMIZADA
# ===========================

# Intentar cargar configuración optimizada
try:
    with open('optimal_config.json', 'r') as f:
        OPTIMAL_CONFIG = json.load(f)
        print("✅ Configuración optimizada cargada desde optimal_config.json")
        
        # Aplicar pesos optimizados
        AGENT_CONFIGS = {
            'momentum': {
                'weight': OPTIMAL_CONFIG['agent_weights']['momentum'],
                'min_confidence': 0.55
            },
            'mean_reversion': {
                'weight': OPTIMAL_CONFIG['agent_weights']['mean_reversion'],
                'min_confidence': 0.55
            },
            'pattern': {
                'weight': OPTIMAL_CONFIG['agent_weights']['pattern'],
                'min_confidence': 0.60
            },
            'volume': {
                'weight': OPTIMAL_CONFIG['agent_weights']['volume'],
                'min_confidence': 0.60
            },
            'sentiment': {
                'weight': OPTIMAL_CONFIG['agent_weights']['sentiment'],
                'min_confidence': 0.65
            }
        }
        
        # Aplicar configuración de consenso
        CONSENSUS_CONFIG = OPTIMAL_CONFIG['consensus']
        
        # Aplicar configuración de riesgo
        RISK_CONFIG = OPTIMAL_CONFIG['risk']
        
except FileNotFoundError:
    print("⚠️ No se encontró optimal_config.json - Usando configuración por defecto")
    
    # Configuración por defecto (mejorada)
    AGENT_CONFIGS = {
        'momentum': {
            'weight': 1.15,      # Momentum ligeramente favorecido
            'min_confidence': 0.55
        },
        'mean_reversion': {
            'weight': 1.0,       # Peso estándar
            'min_confidence': 0.55
        },
        'pattern': {
            'weight': 0.95,      # Ligeramente menos peso
            'min_confidence': 0.60
        },
        'volume': {
            'weight': 1.0,       # Peso estándar
            'min_confidence': 0.60
        },
        'sentiment': {
            'weight': 0.85,      # Menos peso (más ruidoso)
            'min_confidence': 0.65
        }
    }
    
    CONSENSUS_CONFIG = {
        'umbral_consenso_fuerte': 0.55,
        'umbral_consenso_moderado': 0.40,
        'min_agentes_activos': 1,
        'min_confianza_single_agent': 0.80
    }
    
    RISK_CONFIG = {
        'max_position_size': 0.20,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'max_daily_trades': 15
    }

# ===========================
# CONFIGURACIÓN DE TRADING
# ===========================

# Límites generales
MAX_TRADE_SIZE = 50         # Consultar a Claude si trade > $50
SENTIMENT_THRESHOLD = 0.6   # Umbral para señales de sentiment
USE_REAL_BROKER = False     # True para ejecutar en Alpaca, False para solo Supabase
USE_STOP_LOSS = True        # Activar stop loss automático
USE_TRAILING_STOP = True    # Activar trailing stop loss

# Límites de Riesgo (se pueden sobrescribir con RISK_CONFIG)
MAX_DAILY_LOSS = RISK_CONFIG.get('max_daily_loss', 0.05)           # 5% pérdida máxima diaria
MAX_POSITION_SIZE = RISK_CONFIG.get('max_position_size', 0.25)     # 25% máximo por posición
MAX_TOTAL_EXPOSURE = RISK_CONFIG.get('max_total_exposure', 0.80)   # 80% exposición máxima total
STOP_LOSS_PCT = RISK_CONFIG.get('stop_loss_pct', 0.02)             # 2% stop loss
TAKE_PROFIT_PCT = RISK_CONFIG.get('take_profit_pct', 0.03)         # 3% take profit

# Horarios de Trading (hora española)
TRADING_START_HOUR = 15     # 3 PM (9 AM EST)
TRADING_END_HOUR = 22       # 10 PM (4 PM EST)
AVOID_FIRST_HOUR = True     # Evitar primera hora (volatilidad)
AVOID_LAST_HOUR = True      # Evitar última hora

# ===========================
# CONFIGURACIÓN POR SÍMBOLO
# ===========================

SYMBOL_CONFIGS = {
    'COIN': {
        'extra_volatility_factor': 1.2,    # Más volátil por crypto
        'min_volume_ratio': 1.5,
        'correlation_with': 'BTC'
    },
    'MARA': {
        'extra_volatility_factor': 1.3,    # Mining stocks muy volátiles
        'correlation_with': 'BTC',
        'avoid_on_bitcoin_crash': True
    },
    'TSLA': {
        'extra_volatility_factor': 1.1,
        'news_weight': 1.2,                # Muy sensible a noticias
        'elon_tweet_factor': 1.5           # 😄
    },
    'PLTR': {
        'government_contract_boost': 1.2,
        'earnings_volatility': 1.3
    },
    'GME': {
        'meme_factor': 2.0,                # Extrema precaución
        'max_position_override': 0.05,     # Max 5% en meme stocks
        'require_high_confidence': 0.85
    }
}

# ===========================
# CONFIGURACIÓN DE FEATURES ML
# ===========================

ML_FEATURES_CONFIG = {
    'save_features': True,           # Guardar features para ML
    'min_confidence_to_save': 0.6,   # Solo guardar trades con confianza > 60%
    'features_to_extract': [
        'rsi', 'macd', 'bollinger_position',
        'volume_ratio', 'momentum_5d', 'momentum_20d',
        'volatility_20d', 'distance_sma20',
        'fear_greed_index', 'news_sentiment',
        'agent_disagreement', 'consensus_strength'
    ]
}

# ===========================
# CONFIGURACIÓN DE NOTIFICACIONES
# ===========================

NOTIFICATION_CONFIG = {
    'telegram_enabled': bool(TELEGRAM_BOT_TOKEN),
    'discord_enabled': bool(DISCORD_WEBHOOK_URL),
    'email_enabled': bool(EMAIL_USER and EMAIL_PASS),
    'notify_on_trade': True,
    'notify_on_daily_summary': True,
    'notify_on_errors': True,
    'notify_on_high_confidence': 0.8,    # Notificar si confianza > 80%
    'notify_on_large_trade': 100         # Notificar si trade > $100
}

# ===========================
# CONFIGURACIÓN DE BACKTESTING
# ===========================

BACKTEST_CONFIG = {
    'default_lookback_days': 180,
    'walk_forward_days': 30,
    'min_trades_for_validation': 30,
    'transaction_cost': 0.001,           # 0.1% por trade
    'slippage': 0.0005                  # 0.05% slippage
}

# ===========================
# FUNCIONES AUXILIARES
# ===========================

def get_symbols_for_market_condition(volatility_level='normal'):
    """Retorna símbolos apropiados según condición del mercado"""
    if volatility_level == 'high':
        # En alta volatilidad, usar stocks más estables
        return SYMBOLS_CONSERVATIVE
    elif volatility_level == 'low':
        # En baja volatilidad, buscar más oportunidades
        return SYMBOLS_EXTENDED
    else:
        # Condiciones normales
        return SYMBOLS

def get_position_size_for_symbol(symbol, base_size=None):
    """Obtiene tamaño de posición ajustado por símbolo"""
    if base_size is None:
        base_size = MAX_POSITION_SIZE
    
    # Verificar configuración específica del símbolo
    if symbol in SYMBOL_CONFIGS:
        config = SYMBOL_CONFIGS[symbol]
        
        # Override de tamaño máximo
        if 'max_position_override' in config:
            return min(base_size, config['max_position_override'])
        
        # Ajuste por volatilidad extra
        if 'extra_volatility_factor' in config:
            return base_size / config['extra_volatility_factor']
    
    return base_size

def should_trade_symbol(symbol, market_conditions=None):
    """Determina si se debe tradear un símbolo según condiciones"""
    if symbol in SYMBOL_CONFIGS:
        config = SYMBOL_CONFIGS[symbol]
        
        # Verificar condiciones especiales
        if 'avoid_on_bitcoin_crash' in config and market_conditions:
            if market_conditions.get('btc_crash', False):
                return False
        
        if 'require_high_confidence' in config:
            # Este check se hace en el sistema de trading
            pass
    
    return True

# ===========================
# MODO DE OPERACIÓN
# ===========================

# Detectar modo según configuración
if USE_REAL_BROKER and ALPACA_API_KEY and ALPACA_SECRET_KEY:
    TRADING_MODE = 'LIVE' if 'api.alpaca' in ALPACA_BASE_URL else 'PAPER'
else:
    TRADING_MODE = 'SIMULATION'

# ===========================
# VALIDACIÓN DE CONFIGURACIÓN
# ===========================

def validate_configuration():
    """Valida que la configuración sea coherente"""
    issues = []
    
    # Verificar que stop loss < take profit
    if STOP_LOSS_PCT >= TAKE_PROFIT_PCT:
        issues.append("⚠️ Stop loss debe ser menor que take profit")
    
    # Verificar umbrales de consenso
    if CONSENSUS_CONFIG['umbral_consenso_moderado'] >= CONSENSUS_CONFIG['umbral_consenso_fuerte']:
        issues.append("⚠️ Umbral moderado debe ser menor que umbral fuerte")
    
    # Verificar límites de exposición
    if MAX_POSITION_SIZE > MAX_TOTAL_EXPOSURE:
        issues.append("⚠️ Tamaño máximo de posición no puede ser mayor que exposición total")
    
    return issues

# ===========================
# INFORMACIÓN DEL SISTEMA
# ===========================

if __name__ == "__main__":
    print("🔧 CONFIGURACIÓN DEL SISTEMA DE TRADING")
    print("="*60)
    
    # Información básica
    print(f"\n📊 MODO: {TRADING_MODE}")
    print(f"Símbolos principales ({len(SYMBOLS)}): {', '.join(SYMBOLS[:5])}...")
    
    # Conexiones
    print(f"\n🔌 CONEXIONES:")
    print(f"Supabase: {'✅' if SUPABASE_URL and SUPABASE_KEY else '❌'}")
    print(f"Anthropic: {'✅' if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != '' else '❌'}")
    print(f"NewsAPI: {'✅' if NEWSAPI_KEY and NEWSAPI_KEY != '' else '❌'}")
    print(f"Alpaca: {'✅' if ALPACA_API_KEY and ALPACA_SECRET_KEY else '❌'}")
    
    # Notificaciones
    print(f"\n📱 NOTIFICACIONES:")
    print(f"Telegram: {'✅' if NOTIFICATION_CONFIG['telegram_enabled'] else '❌'}")
    print(f"Discord: {'✅' if NOTIFICATION_CONFIG['discord_enabled'] else '❌'}")
    print(f"Email: {'✅' if NOTIFICATION_CONFIG['email_enabled'] else '❌'}")
    
    # Configuración de trading
    print(f"\n⚙️ CONFIGURACIÓN DE TRADING:")
    print(f"Horario: {TRADING_START_HOUR}:00 - {TRADING_END_HOUR}:00")
    print(f"Max posición: {MAX_POSITION_SIZE*100:.0f}%")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.1f}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT*100:.1f}%")
    print(f"Max trades/día: {RISK_CONFIG.get('max_daily_trades', 15)}")
    
    # Pesos de agentes
    print(f"\n🤖 PESOS DE AGENTES:")
    for agent, config in AGENT_CONFIGS.items():
        print(f"  • {agent}: {config['weight']:.2f}x (min conf: {config['min_confidence']})")
    
    # Validación
    issues = validate_configuration()
    if issues:
        print(f"\n⚠️ PROBLEMAS DE CONFIGURACIÓN:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ Configuración validada correctamente")
    
    # Optimización
    if 'OPTIMAL_CONFIG' in globals():
        print(f"\n🎯 Usando configuración optimizada del {OPTIMAL_CONFIG.get('optimized_at', 'N/A')}")
        if 'performance_metrics' in OPTIMAL_CONFIG:
            metrics = OPTIMAL_CONFIG['performance_metrics']
            print(f"   Expected Sharpe: {metrics.get('expected_sharpe', 0):.2f}")
            print(f"   Optimization Score: {metrics.get('optimization_score', 0):.2f}")