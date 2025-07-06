"""
Integración del sistema ML para el trading bot
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Importar el sistema ML
from src.ml.ml_system import MLPredictionSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class MLSignal:
    """Señal de trading del sistema ML"""
    symbol: str
    signal: str
    confidence: float
    price: float
    timestamp: datetime
    rsi: float = 50.0
    features: dict = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
    
    @property
    def should_execute(self) -> bool:
        """Determina si la señal debe ejecutarse"""
        return self.confidence >= 0.70 and self.signal != "HOLD"

class MLTradingBot:
    """
    Bot de trading con ML integrado
    """
    
    def __init__(self, config: Dict = None):
        """Inicializa el bot con ML"""
        self.config = config or self.get_default_config()
        self.ml_system = MLPredictionSystem()
        logger.info("Bot ML inicializado")
    
    @staticmethod
    def get_default_config():
        """Configuración por defecto"""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'min_confidence': 0.70,
            'position_size': 0.1,  # 10% del capital por operación
            'max_positions': 5,
            'stop_loss': 0.02,     # 2% stop loss
            'take_profit': 0.05,   # 5% take profit
        }
    
    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara las características para el ML"""
        df = data.copy()
        
        # Si tiene multi-índice, aplanar
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Calcular indicadores técnicos
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatilidad
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands - CORREGIDO
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_middle'] = bb_middle
        df['BB_upper'] = bb_middle + 2 * bb_std
        df['BB_lower'] = bb_middle - 2 * bb_std
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = np.where(
            df['BB_width'] != 0,
            (df['Close'] - df['BB_lower']) / df['BB_width'],
            0.5
        )
        
        # Volume
        volume_sma = df['Volume'].rolling(20).mean()
        df['Volume_SMA'] = volume_sma
        df['Volume_ratio'] = np.where(
            volume_sma != 0,
            df['Volume'] / volume_sma,
            1.0
        )
        
        return df
    
    def get_ml_signal(self, symbol: str, data: pd.DataFrame) -> Optional[MLSignal]:
        """Obtiene señal ML para un símbolo"""
        try:
            # Preparar datos
            df_features = self.prepare_ml_features(data)
            
            # Tomar última fila válida
            last_row = df_features.dropna().iloc[-1:]
            
            if last_row.empty:
                logger.warning(f"No hay datos válidos para {symbol}")
                return None
            
            # Características para ML
            ml_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'SMA_20', 'SMA_50', 'RSI',
                'Volatility', 'MACD', 'MACD_signal',
                'BB_width', 'BB_position', 'Volume_ratio'
            ]
            
            X = last_row[ml_features]
            
            # Predicción ML
            prediction = self.ml_system.predict(X)
            
            # Crear señal
            signal = MLSignal(
                symbol=symbol,
                signal=prediction['prediction'],
                confidence=prediction['confidence'],
                price=float(last_row['Close'].iloc[0]),
                timestamp=datetime.now(),
                rsi=float(last_row['RSI'].iloc[0]),
                features={
                    'volatility': float(last_row['Volatility'].iloc[0]),
                    'macd': float(last_row['MACD'].iloc[0]),
                    'scaled': prediction.get('scaled', False)
                }
            )
            
            logger.info(f"Señal {symbol}: {signal.signal} @ ${signal.price:.2f} "
                       f"(Confianza: {signal.confidence:.1%})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error obteniendo señal para {symbol}: {e}")
            return None
    
    def analyze_portfolio(self, market_data: Dict[str, pd.DataFrame]) -> List[MLSignal]:
        """Analiza múltiples símbolos y retorna señales"""
        signals = []
        
        for symbol in self.config['symbols']:
            if symbol in market_data:
                signal = self.get_ml_signal(symbol, market_data[symbol])
                if signal and signal.should_execute:
                    signals.append(signal)
        
        # Ordenar por confianza
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def calculate_position_size(self, signal: MLSignal, capital: float) -> float:
        """Calcula el tamaño de la posición"""
        base_size = capital * self.config['position_size']
        
        # Ajustar por confianza
        if signal.confidence >= 0.80:
            size_multiplier = 1.2
        elif signal.confidence >= 0.75:
            size_multiplier = 1.0
        else:
            size_multiplier = 0.8
        
        return base_size * size_multiplier

# Ejemplo de uso
def main():
    """Ejemplo de uso del bot ML"""
    import yfinance as yf
    
    # Crear bot
    bot = MLTradingBot()
    
    # Obtener datos de mercado
    print("\n📊 Obteniendo datos de mercado...")
    market_data = {}
    
    for symbol in bot.config['symbols'][:2]:  # Solo 2 para el ejemplo
        data = yf.download(symbol, period='3mo', interval='1d', progress=False)
        if not data.empty:
            market_data[symbol] = data
            print(f"   ✅ {symbol}: {len(data)} días")
    
    # Analizar portfolio
    print("\n🔍 Analizando señales...")
    signals = bot.analyze_portfolio(market_data)
    
    # Mostrar resultados
    print(f"\n📈 Señales encontradas: {len(signals)}")
    for signal in signals:
        print(f"\n{'='*50}")
        print(f"Símbolo: {signal.symbol}")
        print(f"Señal: {signal.signal}")
        print(f"Precio: ${signal.price:.2f}")
        print(f"Confianza: {signal.confidence:.1%}")
        print(f"RSI: {signal.rsi:.1f}")
        
        # Calcular tamaño de posición
        position = bot.calculate_position_size(signal, 10000)
        print(f"Posición sugerida: ${position:.2f}")

if __name__ == "__main__":
    main()
