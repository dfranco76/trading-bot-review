# Arquitectura del Trading Bot

## 📋 Índice
1. [Visión General](#visión-general)
2. [Componentes Principales](#componentes-principales)
3. [Flujo de Datos](#flujo-de-datos)
4. [Módulos y Responsabilidades](#módulos-y-responsabilidades)
5. [Patrones de Diseño](#patrones-de-diseño)
6. [Configuración](#configuración)
7. [Seguridad](#seguridad)

## 🎯 Visión General

El Trading Bot es un sistema multi-agente automatizado que utiliza múltiples estrategias de trading coordinadas para operar en mercados financieros. La arquitectura está diseñada para ser modular, escalable y resistente a fallos.

### Características Principales:
- **Multi-Estrategia**: 8 agentes especializados trabajando en conjunto
- **Gestión de Riesgo**: Sistema avanzado con circuit breakers
- **Backtesting**: Motor completo para validación de estrategias
- **Monitoreo en Tiempo Real**: Dashboard y sistema de alertas
- **Paper Trading**: Modo práctica sin riesgo real

## 🏗️ Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                        MAIN BOT                              │
│                   (Coordinador Principal)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┬─────────────────┬──────────────┐
    ▼                           ▼                 ▼              ▼
┌──────────┐          ┌──────────────┐   ┌─────────────┐  ┌──────────┐
│STRATEGIES│          │RISK MANAGER  │   │  EXECUTION  │  │MONITORING│
├──────────┤          ├──────────────┤   ├─────────────┤  ├──────────┤
│•Momentum │          │•Risk Limits  │   │•Order Mgmt  │  │•Analytics│
│•Mean Rev │          │•Circuit Break│   │•Broker API  │  │•Alerts   │
│•Pattern  │          │•Position Size│   │•Trade Exec  │  │•Reports  │
│•Sentiment│          │•Stop Loss    │   │•Slippage    │  │•Metrics  │
│•Volume   │          └──────────────┘   └─────────────┘  └──────────┘
│•ML Pred  │                                      
└──────────┘                                      
```

## 📊 Flujo de Datos

### 1. **Flujo Principal de Trading**
```python
Market Data → Data Module → Strategies → Risk Manager → Execution → Market
     ↓                          ↓              ↓             ↓
  Monitoring ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
```

### 2. **Proceso de Decisión**
1. **Recolección de Datos**: El módulo de datos obtiene información del mercado
2. **Análisis Multi-Agente**: Cada estrategia analiza los datos independientemente
3. **Consenso**: El sistema multi-agente pondera las señales
4. **Validación de Riesgo**: Risk Manager valida la operación
5. **Ejecución**: Si pasa validación, se ejecuta la orden
6. **Monitoreo**: Se registran métricas y se generan alertas si es necesario

## 🗂️ Módulos y Responsabilidades

### `/src/strategies/`
Contiene todos los agentes de trading y el sistema coordinador.

#### **sistema_multiagente.py**
- **Responsabilidad**: Coordinar todos los agentes y generar señales consolidadas
- **Entradas**: Datos de mercado, configuración
- **Salidas**: Señales de trading ponderadas
- **Dependencias**: Todos los agentes individuales

#### **Agentes Especializados**:
1. **agente_momentum.py**: Detecta tendencias fuertes
2. **agente_mean_reversion.py**: Identifica reversiones a la media
3. **agente_pattern_recognition.py**: Reconoce patrones técnicos
4. **agente_sentiment.py**: Analiza sentimiento del mercado
5. **agente_volume_momentum.py**: Señales basadas en volumen
6. **ml_prediction_system.py**: Predicciones con Machine Learning
7. **portfolio_optimization_system.py**: Optimización de portfolio

### `/src/risk_management/`
Gestión integral del riesgo.

#### **risk_manager.py (EnhancedRiskManager)**
- **Funciones**:
  - Validación de tamaño de posición
  - Cálculo de stop-loss dinámico
  - Gestión de exposición total
  - Validación de correlaciones

#### **circuit_breaker_system.py**
- **Funciones**:
  - Detención automática por pérdidas
  - Límites de operaciones diarias
  - Protección contra volatilidad extrema

### `/src/execution/`
Manejo de órdenes y conexión con brokers.

#### **trade_manager.py**
- **Responsabilidades**:
  - Gestión del ciclo de vida de las órdenes
  - Registro de operaciones
  - Cálculo de slippage

#### **broker_integration.py**
- **Funciones**:
  - APIs de diferentes brokers
  - Manejo de reconexiones
  - Validación de órdenes

#### **advanced_order_execution.py**
- **Características**:
  - Órdenes iceberg
  - TWAP/VWAP
  - Smart routing

### `/src/monitoring/`
Sistema completo de monitoreo y análisis.

#### **market_analyzer.py**
- Análisis en tiempo real del mercado
- Detección de condiciones anormales

#### **automated_reporting_system.py**
- Generación automática de reportes
- Métricas de performance

#### **notification_system.py**
- Alertas por email/SMS/Telegram
- Notificaciones críticas

### `/src/backtesting/`
Validación histórica de estrategias.

#### **backtesting_engine.py**
- Motor de simulación histórica
- Cálculo de métricas (Sharpe, Drawdown, etc.)

### `/src/data/`
Gestión de datos del mercado.

#### **demo_analysis.py**
- Análisis de datos para demos
- Generación de datos sintéticos

### `/src/utils/`
Utilidades y configuración.

#### **config.py**
- Gestión centralizada de configuración
- Variables de entorno

#### **safe_trading.py**
- Funciones de seguridad
- Validaciones críticas

## 🎨 Patrones de Diseño

### 1. **Strategy Pattern**
Cada agente implementa una estrategia independiente con interfaz común:
```python
class BaseAgent:
    def analyze(self, market_data) -> Signal:
        pass
    
    def get_confidence(self) -> float:
        pass
```

### 2. **Observer Pattern**
El sistema de monitoreo observa todos los componentes:
```python
class TradingEventObserver:
    def on_signal_generated(self, signal):
        # Log, alert, update dashboard
        pass
```

### 3. **Factory Pattern**
Creación de órdenes según el tipo:
```python
class OrderFactory:
    def create_order(self, order_type, params):
        if order_type == "MARKET":
            return MarketOrder(params)
        elif order_type == "LIMIT":
            return LimitOrder(params)
```

### 4. **Singleton Pattern**
Configuración y conexiones de broker:
```python
class BrokerConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

## ⚙️ Configuración

### **optimal_config.json**
```json
{
    "trading": {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "5m",
        "max_positions": 3
    },
    "risk": {
        "max_position_size": 0.1,
        "stop_loss_percentage": 0.02,
        "daily_loss_limit": 0.05
    },
    "agents": {
        "momentum": {"weight": 0.2, "enabled": true},
        "mean_reversion": {"weight": 0.2, "enabled": true},
        "ml_prediction": {"weight": 0.3, "enabled": true}
    }
}
```

### **Variables de Entorno (.env)**
```bash
# Exchange API Keys
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Database
DATABASE_URL=postgresql://user:pass@localhost/trading_bot

# Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Mode
TRADING_MODE=paper  # paper|live
```

## 🔒 Seguridad

### 1. **Gestión de API Keys**
- Nunca en código fuente
- Uso de variables de entorno
- Rotación periódica

### 2. **Validaciones**
- Doble validación de órdenes
- Límites estrictos de posición
- Circuit breakers automáticos

### 3. **Logging y Auditoría**
- Registro de todas las operaciones
- Trazabilidad completa
- Alertas de actividad sospechosa

### 4. **Modo Paper Trading**
- Pruebas sin riesgo real
- Validación de estrategias
- Entrenamiento de nuevos usuarios

## 🚀 Deployment

### **Requisitos del Sistema**
- Python 3.8+
- PostgreSQL 12+
- Redis (para caché)
- 4GB RAM mínimo
- Conexión estable a Internet

### **Instalación**
```bash
# Clonar repositorio
git clone https://github.com/usuario/trading-bot.git

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Ejecutar migraciones de BD
python manage.py migrate

# Ejecutar bot
python src/main_bot.py
```

## 📈 Métricas de Performance

El sistema rastrea automáticamente:
- **ROI** (Return on Investment)
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Win Rate**
- **Average Trade Duration**
- **Profit Factor**

## 🔄 Actualizaciones y Mantenimiento

### **Actualización de Estrategias**
1. Desarrollar nueva estrategia heredando de `BaseAgent`
2. Registrar en `sistema_multiagente.py`
3. Configurar peso en `optimal_config.json`
4. Backtest exhaustivo antes de activar

### **Monitoreo de Salud**
- Logs en `/logs/`
- Métricas en tiempo real en dashboard
- Alertas automáticas por anomalías

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear branch para feature (`git checkout -b feature/NuevaEstrategia`)
3. Commit cambios (`git commit -m 'Add: Nueva estrategia X'`)
4. Push al branch (`git push origin feature/NuevaEstrategia`)
5. Crear Pull Request

## 📞 Soporte

- **Issues**: GitHub Issues
- **Email**: soporte@tradingbot.com
- **Documentación**: `/docs/`