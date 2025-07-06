# ai_evolution_plan.py - Roadmap para evolucionar hacia IA

"""
FASE 1: PREPARACIÓN DE DATOS (Mes 1-3)
Mientras el bot opera con reglas, preparamos los datos para ML
"""

# 1. data_collector.py - Enriquecer datos actuales
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from supabase import create_client

class DataCollector:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def create_ml_tables(self):
        """Crear nuevas tablas para ML en Supabase"""
        
        # Tabla de features para cada decisión
        """
        CREATE TABLE ml_features (
            id SERIAL PRIMARY KEY,
            trade_id INTEGER REFERENCES trades(id),
            symbol VARCHAR(10),
            timestamp TIMESTAMP,
            
            -- Features técnicos
            rsi FLOAT,
            macd FLOAT,
            macd_signal FLOAT,
            bollinger_position FLOAT,
            volume_ratio FLOAT,
            price_momentum_1d FLOAT,
            price_momentum_5d FLOAT,
            price_momentum_20d FLOAT,
            volatility_20d FLOAT,
            
            -- Features de mercado
            spy_correlation FLOAT,
            sector_momentum FLOAT,
            market_regime VARCHAR(20), -- trending/ranging/volatile
            
            -- Features de sentiment
            news_sentiment FLOAT,
            social_volume FLOAT,
            fear_greed_index FLOAT,
            
            -- Label (resultado real)
            price_change_1h FLOAT,
            price_change_4h FLOAT,
            price_change_24h FLOAT,
            actual_best_action VARCHAR(10), -- calculado después
            
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
    def collect_features_for_trade(self, symbol, decision_data):
        """Recolecta todas las features en el momento de la decisión"""
        
        features = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            
            # Extraer de los agentes actuales
            'rsi': self.get_rsi(symbol),
            'macd': self.get_macd(symbol),
            'volume_ratio': self.get_volume_ratio(symbol),
            
            # Calcular adicionales
            'spy_correlation': self.calculate_spy_correlation(symbol),
            'market_regime': self.detect_market_regime(),
            
            # Guardar para calcular label después
            'entry_price': decision_data['price']
        }
        
        # Insertar en base de datos
        self.supabase.table('ml_features').insert(features).execute()
        
        # Programar actualización del label en 1h, 4h, 24h
        self.schedule_label_update(features['id'])


"""
FASE 2: FEATURE ENGINEERING (Mes 3-4)
Crear features más sofisticados
"""

# 2. feature_engineering.py
class FeatureEngineer:
    def __init__(self):
        self.feature_list = []
        
    def create_advanced_features(self, df):
        """Crea features avanzados para ML"""
        
        # 1. Features de microestructura
        df['bid_ask_spread'] = self.calculate_spread(df)
        df['order_flow_imbalance'] = self.calculate_order_flow(df)
        
        # 2. Features estadísticos
        df['price_kurtosis'] = df['Close'].rolling(20).apply(lambda x: x.kurtosis())
        df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        # 3. Features de patrones
        df['is_doji'] = self.detect_doji(df)
        df['is_hammer'] = self.detect_hammer(df)
        df['support_distance'] = self.distance_to_support(df)
        
        # 4. Features de régimen de mercado
        df['trend_strength'] = self.calculate_trend_strength(df)
        df['choppiness_index'] = self.calculate_choppiness(df)
        
        # 5. Features inter-mercado
        df['dxy_correlation'] = self.correlate_with_dxy(df)  # Dollar index
        df['vix_level'] = self.get_vix_level()
        
        # 6. Features de tiempo
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['days_to_earnings'] = self.days_to_next_earnings(df)
        
        # 7. Features de los agentes (meta-features)
        df['agent_disagreement'] = self.calculate_agent_disagreement()
        df['consensus_strength'] = self.calculate_consensus_strength()
        
        return df
    
    def create_labels(self, df, horizon='1h'):
        """Crea labels para aprendizaje supervisado"""
        
        if horizon == '1h':
            shift = 12  # 12 periodos de 5 min = 1 hora
        elif horizon == '4h':
            shift = 48
        elif horizon == '1d':
            shift = 288
            
        # Clasificación (subida/bajada)
        df['future_return'] = df['Close'].shift(-shift) / df['Close'] - 1
        df['label_classification'] = np.where(df['future_return'] > 0.002, 1,  # Sube >0.2%
                                            np.where(df['future_return'] < -0.002, -1,  # Baja >0.2%
                                                    0))  # Lateral
        
        # Regresión (magnitud del movimiento)
        df['label_regression'] = df['future_return']
        
        # Trading óptimo (más complejo)
        df['optimal_action'] = self.calculate_optimal_action(df, shift)
        
        return df


"""
FASE 3: MODELOS DE ML (Mes 4-6)
Empezar con modelos simples
"""

# 3. ml_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

class MLModels:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def train_classification_models(self, X, y):
        """Entrena modelos de clasificación"""
        
        # Split temporal (nunca usar datos futuros)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 1. Random Forest (robusto, interpretable)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,  # Evitar overfitting
            random_state=42
        )
        
        # 2. XGBoost (mejor performance)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            objective='multi:softprob',
            random_state=42
        )
        
        # 3. LightGBM (más rápido)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            objective='multiclass',
            random_state=42
        )
        
        # Validación cruzada temporal
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar modelos
            rf.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            lgb_model.fit(X_train, y_train)
            
            # Evaluar
            print(f"RF Score: {rf.score(X_val, y_val)}")
            print(f"XGB Score: {xgb_model.score(X_val, y_val)}")
            print(f"LGB Score: {lgb_model.score(X_val, y_val)}")
        
        # Guardar modelos
        self.models['rf'] = rf
        self.models['xgb'] = xgb_model
        self.models['lgb'] = lgb_model
        
        # Feature importance
        self.analyze_feature_importance()
    
    def create_ensemble(self):
        """Crea ensemble de modelos"""
        
        class EnsembleModel:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights or [1/len(models)] * len(models)
                
            def predict_proba(self, X):
                predictions = []
                for model, weight in zip(self.models, self.weights):
                    pred = model.predict_proba(X)
                    predictions.append(pred * weight)
                
                return np.sum(predictions, axis=0)
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1)
        
        return EnsembleModel(list(self.models.values()))


"""
FASE 4: DEEP LEARNING (Mes 6-9)
Modelos más complejos
"""

# 4. deep_learning_models.py
import tensorflow as tf
from tensorflow.keras import layers, models

class DeepLearningModels:
    def __init__(self):
        self.models = {}
        
    def build_lstm_model(self, input_shape):
        """LSTM para series temporales"""
        
        model = models.Sequential([
            # LSTM layers
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')  # Buy/Hold/Sell
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """CNN-LSTM para patrones + secuencias"""
        
        model = models.Sequential([
            # CNN para extraer patrones locales
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 3, activation='relu'),
            
            # LSTM para patrones temporales
            layers.LSTM(50, return_sequences=True),
            layers.LSTM(25),
            
            # Output
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        return model
    
    def build_attention_model(self, input_shape):
        """Transformer-based model"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(inputs, inputs)
        
        # Add & Norm
        x = layers.LayerNormalization()(inputs + attention)
        
        # Feed forward
        ff = layers.Dense(128, activation='relu')(x)
        ff = layers.Dense(input_shape[-1])(ff)
        
        # Add & Norm
        x = layers.LayerNormalization()(x + ff)
        
        # Classification head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model


"""
FASE 5: REINFORCEMENT LEARNING (Mes 9-12)
Trading como un juego
"""

# 5. reinforcement_learning.py
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, A2C, SAC

class TradingEnvironment(gym.Env):
    """Trading environment para RL"""
    
    def __init__(self, df, initial_balance=200):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: todas las features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(df.columns),),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action):
        # Ejecutar acción
        current_price = self.df.iloc[self.current_step]['Close']
        
        if action == 1 and self.balance > 0:  # Buy
            shares = self.balance / current_price
            self.position += shares
            self.balance = 0
            self.trades.append(('BUY', current_price, shares))
            
        elif action == 2 and self.position > 0:  # Sell
            self.balance = self.position * current_price
            self.trades.append(('SELL', current_price, self.position))
            self.position = 0
        
        # Calcular reward
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
            reward = self._calculate_final_reward()
        else:
            done = False
            reward = self._calculate_step_reward()
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_step_reward(self):
        """Reward por cada paso"""
        # Portfolio value
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price
        
        # Reward basado en cambio de valor
        if hasattr(self, 'previous_portfolio_value'):
            reward = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        else:
            reward = 0
            
        self.previous_portfolio_value = portfolio_value
        
        return reward
    
    def _calculate_final_reward(self):
        """Reward final con métricas adicionales"""
        current_price = self.df.iloc[self.current_step]['Close']
        final_value = self.balance + self.position * current_price
        
        # Return total
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio simplificado
        returns = []
        for i in range(1, len(self.trades)):
            if self.trades[i][0] == 'SELL' and self.trades[i-1][0] == 'BUY':
                trade_return = (self.trades[i][1] - self.trades[i-1][1]) / self.trades[i-1][1]
                returns.append(trade_return)
        
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
        else:
            sharpe = 0
        
        # Reward combinado
        reward = total_return + 0.1 * sharpe
        
        return reward


"""
FASE 6: INTEGRACIÓN CON SISTEMA ACTUAL (Mes 12+)
Nuevo agente que usa ML
"""

# 6. agente_ml.py
class AgenteML:
    def __init__(self):
        self.nombre = "Agente ML"
        self.models = self.load_models()
        self.feature_engineer = FeatureEngineer()
        self.min_confidence_threshold = 0.65
        
    def load_models(self):
        """Carga modelos entrenados"""
        models = {
            'rf': joblib.load('models/random_forest_latest.pkl'),
            'xgb': joblib.load('models/xgboost_latest.pkl'),
            'lstm': tf.keras.models.load_model('models/lstm_latest.h5'),
            'rl_agent': PPO.load('models/ppo_trader_latest')
        }
        return models
    
    def analyze_symbol(self, symbol):
        """Análisis usando ensemble de ML"""
        
        # 1. Obtener datos
        data = yf.download(symbol, period='2mo', interval='5m')
        
        # 2. Feature engineering
        features = self.feature_engineer.create_advanced_features(data)
        latest_features = features.iloc[-1]
        
        # 3. Predicciones de cada modelo
        predictions = {}
        
        # Random Forest & XGBoost
        for name in ['rf', 'xgb']:
            model = self.models[name]
            pred_proba = model.predict_proba([latest_features])[0]
            predictions[name] = {
                'buy': pred_proba[1],
                'sell': pred_proba[2],
                'hold': pred_proba[0]
            }
        
        # LSTM (necesita secuencia)
        sequence = features.iloc[-50:].values.reshape(1, 50, -1)
        lstm_pred = self.models['lstm'].predict(sequence)[0]
        predictions['lstm'] = {
            'buy': lstm_pred[1],
            'sell': lstm_pred[2],
            'hold': lstm_pred[0]
        }
        
        # RL Agent
        rl_action = self.models['rl_agent'].predict(latest_features)[0]
        predictions['rl'] = {
            'buy': 1.0 if rl_action == 1 else 0.0,
            'sell': 1.0 if rl_action == 2 else 0.0,
            'hold': 1.0 if rl_action == 0 else 0.0
        }
        
        # 4. Ensemble - Promedio ponderado
        weights = {'rf': 0.2, 'xgb': 0.3, 'lstm': 0.3, 'rl': 0.2}
        
        ensemble_pred = {
            'buy': sum(predictions[m]['buy'] * weights[m] for m in weights),
            'sell': sum(predictions[m]['sell'] * weights[m] for m in weights),
            'hold': sum(predictions[m]['hold'] * weights[m] for m in weights)
        }
        
        # 5. Decisión final
        max_action = max(ensemble_pred, key=ensemble_pred.get)
        confidence = ensemble_pred[max_action]
        
        # 6. Explicabilidad - Por qué tomó esa decisión
        feature_importance = self.get_feature_importance(latest_features)
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        reason = f"ML Ensemble: "
        reason += f"{max_action.upper()} con {confidence:.0%} confianza. "
        reason += f"Factores clave: {', '.join([f[0] for f in top_features])}"
        
        # Solo dar señal si la confianza es alta
        if confidence < self.min_confidence_threshold:
            max_action = 'hold'
            confidence = 0.5
            reason = f"Confianza insuficiente ({confidence:.0%})"
        
        return {
            'action': max_action.upper(),
            'confidence': confidence,
            'price': data['Close'].iloc[-1],
            'reason': reason,
            'ml_predictions': predictions  # Para debugging
        }
    
    def get_feature_importance(self, features):
        """Obtiene importancia de features para explicabilidad"""
        # Usar SHAP o feature importance del modelo
        rf_importance = self.models['rf'].feature_importances_
        feature_names = features.index
        
        importance_dict = {}
        for name, imp in zip(feature_names, rf_importance):
            importance_dict[name] = imp
            
        return importance_dict


"""
FASE 7: MONITOREO Y MEJORA CONTINUA
Sistema de feedback loop
"""

# 7. ml_monitor.py
class MLMonitor:
    def __init__(self):
        self.performance_history = []
        
    def evaluate_model_performance(self):
        """Evalúa performance de modelos en producción"""
        
        # Obtener predicciones vs resultados reales
        query = """
        SELECT 
            mf.symbol,
            mf.timestamp,
            mf.ml_prediction,
            mf.ml_confidence,
            mf.actual_best_action,
            t.pnl
        FROM ml_features mf
        JOIN trades t ON mf.trade_id = t.id
        WHERE mf.timestamp > NOW() - INTERVAL '7 days'
        """
        
        results = pd.read_sql(query, connection)
        
        # Métricas
        accuracy = (results['ml_prediction'] == results['actual_best_action']).mean()
        
        # Profit factor
        ml_trades = results[results['ml_confidence'] > 0.65]
        profit_factor = ml_trades[ml_trades['pnl'] > 0]['pnl'].sum() / abs(ml_trades[ml_trades['pnl'] < 0]['pnl'].sum())
        
        # Guardar métricas
        self.performance_history.append({
            'date': datetime.now(),
            'accuracy': accuracy,
            'profit_factor': profit_factor,
            'total_trades': len(ml_trades)
        })
        
        # Alertas si performance baja
        if accuracy < 0.45:
            self.send_alert("ML Accuracy below 45%!")
        
        return {
            'accuracy': accuracy,
            'profit_factor': profit_factor,
            'should_retrain': accuracy < 0.5
        }
    
    def auto_retrain(self):
        """Re-entrena modelos automáticamente"""
        
        # Obtener datos más recientes
        recent_data = self.get_recent_training_data()
        
        # Re-entrenar solo si hay suficientes datos nuevos
        if len(recent_data) > 1000:
            ml_models = MLModels()
            ml_models.train_classification_models(
                recent_data.drop(['label'], axis=1),
                recent_data['label']
            )
            
            # Guardar nuevos modelos
            self.save_new_models(ml_models)
            
            # A/B testing: comparar modelo nuevo vs antiguo
            self.run_ab_test()


# Función principal para ejecutar todo
def main():
    """Pipeline completo de ML"""
    
    # 1. Recolectar datos
    collector = DataCollector()
    collector.collect_historical_data()
    
    # 2. Feature engineering
    engineer = FeatureEngineer()
    features = engineer.create_all_features()
    
    # 3. Entrenar modelos
    ml = MLModels()
    ml.train_all_models(features)
    
    # 4. Backtest
    backtester = Backtester()
    results = backtester.run_backtest(ml.models)
    
    # 5. Deploy si es bueno
    if results['sharpe_ratio'] > 1.5:
        deploy_model(ml.models['ensemble'])
    
    print("ML Pipeline completado!")


if __name__ == "__main__":
    main()