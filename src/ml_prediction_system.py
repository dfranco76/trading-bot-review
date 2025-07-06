# ml_prediction_system.py
"""
Sistema completo de Machine Learning para predicción de trades
Incluye feature engineering, múltiples modelos y validación
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
import os
warnings.filterwarnings('ignore')

# Intentar importar librerías avanzadas
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Instalar con: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM no disponible. Instalar con: pip install lightgbm")

class MLPredictionSystem:
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.ensemble_weights = {}
        
        # Configuración de features
        self.feature_groups = {
            'price_features': [
                'rsi', 'momentum_5d', 'momentum_20d', 'distance_sma20',
                'volatility_20d', 'high_low_ratio', 'close_position'
            ],
            'volume_features': [
                'volume_ratio', 'volume_trend', 'obv_slope', 'mfi'
            ],
            'pattern_features': [
                'support_distance', 'resistance_distance', 'trend_strength',
                'price_acceleration', 'volatility_change'
            ],
            'consensus_features': [
                'votes_buy', 'votes_sell', 'votes_hold', 'avg_confidence_buy',
                'avg_confidence_sell', 'consensus_confidence', 'unanimity'
            ],
            'agent_features': [
                'momentum_conf', 'mean_rev_conf', 'pattern_conf',
                'volume_conf', 'sentiment_conf'
            ],
            'market_features': [
                'market_regime_bull', 'market_regime_bear', 'market_volatility',
                'sector_momentum', 'correlation_avg'
            ],
            'temporal_features': [
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                'minutes_since_open_normalized'
            ]
        }
        
        # Modelos a entrenar
        self.model_configs = {
            'logistic': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'weight': 0.15
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'weight': 0.20
            },
            'gradient_boost': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'weight': 0.20
            },
            'neural_net': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                ),
                'weight': 0.15
            }
        }
        
        # Añadir modelos avanzados si están disponibles
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective='multi:softprob',
                    random_state=42,
                    use_label_encoder=False
                ),
                'weight': 0.15
            }
        
        if LIGHTGBM_AVAILABLE:
            self.model_configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                ),
                'weight': 0.15
            }
        
        # Normalizar pesos
        total_weight = sum(config['weight'] for config in self.model_configs.values())
        for model_name in self.model_configs:
            self.model_configs[model_name]['weight'] /= total_weight
        
        print(f"🤖 ML Prediction System iniciado con {len(self.model_configs)} modelos")
    
    def engineer_features(self, raw_features: Dict) -> pd.DataFrame:
        """Ingeniería de features avanzada"""
        features = {}
        
        # 1. Features de precio
        features['rsi'] = raw_features.get('rsi', 50) / 100
        features['momentum_5d'] = np.clip(raw_features.get('momentum_5d', 0), -0.2, 0.2)
        features['momentum_20d'] = np.clip(raw_features.get('momentum_20d', 0), -0.3, 0.3)
        features['distance_sma20'] = np.clip(raw_features.get('distance_sma20', 0) / 100, -0.2, 0.2)
        features['volatility_20d'] = np.clip(raw_features.get('volatility_20d', 0.02), 0, 0.1)
        features['high_low_ratio'] = raw_features.get('high_low_ratio', 0.02)
        features['close_position'] = raw_features.get('close_position', 0.5)
        
        # 2. Features de volumen
        features['volume_ratio'] = np.log1p(raw_features.get('volume_ratio', 1))
        features['volume_trend'] = raw_features.get('volume_trend', 0)
        features['obv_slope'] = np.clip(raw_features.get('obv_slope', 0), -1, 1)
        features['mfi'] = raw_features.get('mfi', 50) / 100
        
        # 3. Features de patrones técnicos
        features['support_distance'] = raw_features.get('support_distance', 0)
        features['resistance_distance'] = raw_features.get('resistance_distance', 0)
        features['trend_strength'] = raw_features.get('trend_strength', 0)
        features['price_acceleration'] = raw_features.get('price_acceleration', 0)
        features['volatility_change'] = raw_features.get('volatility_change', 0)
        
        # 4. Features de consenso
        features['votes_buy'] = raw_features.get('votes_buy', 0) / 5
        features['votes_sell'] = raw_features.get('votes_sell', 0) / 5
        features['votes_hold'] = raw_features.get('votes_hold', 0) / 5
        features['avg_confidence_buy'] = raw_features.get('avg_confidence_buy', 0)
        features['avg_confidence_sell'] = raw_features.get('avg_confidence_sell', 0)
        features['consensus_confidence'] = raw_features.get('consensus_confidence', 0)
        features['unanimity'] = raw_features.get('unanimity', 0)
        
        # 5. Features de agentes individuales
        features['momentum_conf'] = raw_features.get('momentum_conf', 0.5)
        features['mean_rev_conf'] = raw_features.get('mean_rev_conf', 0.5)
        features['pattern_conf'] = raw_features.get('pattern_conf', 0.5)
        features['volume_conf'] = raw_features.get('volume_conf', 0.5)
        features['sentiment_conf'] = raw_features.get('sentiment_conf', 0.5)
        
        # 6. Features de mercado
        market_regime = raw_features.get('market_regime', 'unknown')
        features['market_regime_bull'] = 1 if market_regime == 'bull' else 0
        features['market_regime_bear'] = 1 if market_regime == 'bear' else 0
        features['market_volatility'] = raw_features.get('market_volatility', 0.02)
        features['sector_momentum'] = raw_features.get('sector_momentum', 0)
        features['correlation_avg'] = raw_features.get('correlation_avg', 0.3)
        
        # 7. Features temporales (cíclicas)
        hour = raw_features.get('hour', 12)
        day_of_week = raw_features.get('day_of_week', 2)
        
        # Encoding cíclico para hora
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Encoding cíclico para día de la semana
        features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        features['minutes_since_open_normalized'] = raw_features.get('minutes_since_open', 0) / 390
        
        # 8. Features de interacción
        features['momentum_volume_interaction'] = features['momentum_5d'] * features['volume_ratio']
        features['confidence_unanimity_interaction'] = features['consensus_confidence'] * features['unanimity']
        features['volatility_regime_interaction'] = features['volatility_20d'] * (features['market_regime_bear'] + 0.5)
        
        # 9. Features de ratio
        if features['votes_sell'] > 0:
            features['buy_sell_ratio'] = features['votes_buy'] / features['votes_sell']
        else:
            features['buy_sell_ratio'] = features['votes_buy'] * 5
        
        features['confidence_spread'] = features['avg_confidence_buy'] - features['avg_confidence_sell']
        
        # 10. Features polinómicas para relaciones no lineales
        features['rsi_squared'] = features['rsi'] ** 2
        features['momentum_abs'] = abs(features['momentum_5d'])
        features['volatility_sqrt'] = np.sqrt(features['volatility_20d'])
        
        return pd.DataFrame([features])
    
    def prepare_training_data(self, lookback_days: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara datos de entrenamiento desde la base de datos"""
        if not self.supabase:
            print("⚠️ No hay conexión a Supabase")
            return pd.DataFrame(), pd.Series()
        
        try:
            # Obtener features históricas con labels
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            response = self.supabase.table('ml_features')\
                .select("*")\
                .gte('timestamp', cutoff_date.isoformat())\
                .not_.is_('actual_best_action', None)\
                .execute()
            
            if not response.data:
                print("⚠️ No hay datos de entrenamiento disponibles")
                return pd.DataFrame(), pd.Series()
            
            # Procesar datos
            X_list = []
            y_list = []
            
            for record in response.data:
                # Parsear features
                raw_features = json.loads(record['features'])
                
                # Añadir información de mercado si está disponible
                raw_features['market_regime'] = record.get('market_regime', 'unknown')
                raw_features['market_volatility'] = record.get('market_volatility', 0.02)
                
                # Ingenería de features
                features_df = self.engineer_features(raw_features)
                X_list.append(features_df)
                
                # Label
                y_list.append(record['actual_best_action'])
            
            # Combinar todos los datos
            X = pd.concat(X_list, ignore_index=True)
            y = pd.Series(y_list)
            
            # Imputar valores faltantes
            X = X.fillna(X.median())
            
            print(f"✅ Preparados {len(X)} ejemplos de entrenamiento")
            print(f"   Distribución: BUY={sum(y=='BUY')}, SELL={sum(y=='SELL')}, HOLD={sum(y=='HOLD')}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparando datos: {e}")
            return pd.DataFrame(), pd.Series()
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> List[str]:
        """Selección de features usando múltiples métodos"""
        print("\n🔍 Seleccionando mejores features...")
        
        # Método 1: SelectKBest con ANOVA F-test
        selector_anova = SelectKBest(score_func=f_classif, k=min(n_features, len(X.columns)))
        selector_anova.fit(X, y)
        scores_anova = pd.Series(selector_anova.scores_, index=X.columns)
        top_anova = scores_anova.nlargest(n_features).index.tolist()
        
        # Método 2: Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance_rf = pd.Series(rf.feature_importances_, index=X.columns)
        top_rf = importance_rf.nlargest(n_features).index.tolist()
        
        # Método 3: RFE con Logistic Regression
        if len(X) > 100:  # Solo si hay suficientes datos
            lr = LogisticRegression(random_state=42, max_iter=100)
            rfe = RFE(lr, n_features_to_select=n_features)
            rfe.fit(X, y)
            top_rfe = X.columns[rfe.support_].tolist()
        else:
            top_rfe = []
        
        # Combinar resultados (voting)
        all_features = set(top_anova + top_rf + top_rfe)
        feature_votes = {}
        
        for feature in all_features:
            votes = 0
            if feature in top_anova:
                votes += 1
            if feature in top_rf:
                votes += 1
            if feature in top_rfe:
                votes += 1
            feature_votes[feature] = votes
        
        # Seleccionar features con más votos
        selected_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in selected_features[:n_features]]
        
        # Asegurar que tenemos features esenciales
        essential_features = [
            'consensus_confidence', 'votes_buy', 'votes_sell',
            'rsi', 'momentum_5d', 'volume_ratio'
        ]
        
        for feature in essential_features:
            if feature not in selected_features and feature in X.columns:
                selected_features.append(feature)
        
        # Limitar a n_features
        selected_features = selected_features[:n_features]
        
        print(f"✅ Seleccionadas {len(selected_features)} features")
        print(f"   Top 10: {selected_features[:10]}")
        
        return selected_features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str] = None):
        """Entrena todos los modelos con validación cruzada"""
        if len(X) < 50:
            print("⚠️ Datos insuficientes para entrenamiento")
            return
        
        # Usar features seleccionadas o todas
        if selected_features:
            X = X[selected_features]
        
        # Guardar lista de features
        self.selected_features = X.columns.tolist()
        
        # Escalar datos
        self.scalers['robust'] = RobustScaler()
        X_scaled = self.scalers['robust'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Time series split para validación
        tscv = TimeSeriesSplit(n_splits=3)
        
        print("\n🚀 Entrenando modelos...")
        
        for model_name, config in self.model_configs.items():
            print(f"\n📊 Entrenando {model_name}...")
            
            try:
                model = config['model']
                
                # Validación cruzada
                scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=tscv, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                print(f"   Accuracy CV: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
                
                # Entrenar modelo completo
                model.fit(X_scaled, y)
                self.models[model_name] = model
                
                # Guardar métricas
                self.performance_metrics[model_name] = {
                    'cv_accuracy': scores.mean(),
                    'cv_std': scores.std(),
                    'trained_samples': len(X)
                }
                
                # Feature importance (si está disponible)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = pd.Series(
                        model.feature_importances_,
                        index=X.columns
                    ).sort_values(ascending=False)
                
            except Exception as e:
                print(f"   ❌ Error entrenando {model_name}: {e}")
        
        # Entrenar meta-modelo para ensemble
        self._train_ensemble(X_scaled, y, tscv)
        
        print("\n✅ Entrenamiento completado")
    
    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series, cv):
        """Entrena ensemble con stacking"""
        print("\n🔄 Entrenando ensemble...")
        
        # Obtener predicciones de cada modelo
        predictions = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                predictions[f'{model_name}_buy'] = pred_proba[:, 0]
                predictions[f'{model_name}_sell'] = pred_proba[:, 1]
                if pred_proba.shape[1] > 2:
                    predictions[f'{model_name}_hold'] = pred_proba[:, 2]
        
        if not predictions:
            print("⚠️ No hay predicciones para ensemble")
            return
        
        # Crear dataframe de meta-features
        meta_features = pd.DataFrame(predictions)
        
        # Entrenar meta-modelo simple
        self.ensemble_model = LogisticRegression(random_state=42)
        self.ensemble_model.fit(meta_features, y)
        
        print("✅ Ensemble entrenado")
    
    def predict(self, raw_features: Dict) -> Dict:
        """Realiza predicción con todos los modelos"""
        if not self.models:
            return {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'model_predictions': {},
                'explanation': 'Modelos no entrenados'
            }
        
        try:
            # Preparar features
            features_df = self.engineer_features(raw_features)
            
            # Seleccionar features entrenadas
            if hasattr(self, 'selected_features'):
                features_df = features_df[self.selected_features]
            
            # Escalar
            features_scaled = self.scalers['robust'].transform(features_df)
            
            # Predicciones individuales
            model_predictions = {}
            model_confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    model_predictions[model_name] = pred
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        model_confidences[model_name] = proba.max()
                    else:
                        model_confidences[model_name] = 0.5
                        
                except Exception as e:
                    print(f"Error prediciendo con {model_name}: {e}")
                    model_predictions[model_name] = 'HOLD'
                    model_confidences[model_name] = 0.5
            
            # Votación ponderada
            action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for model_name, prediction in model_predictions.items():
                weight = self.model_configs[model_name]['weight']
                confidence = model_confidences[model_name]
                action_scores[prediction] += weight * confidence
            
            # Predicción final
            final_prediction = max(action_scores.items(), key=lambda x: x[1])[0]
            final_confidence = action_scores[final_prediction] / sum(action_scores.values())
            
            # Generar explicación
            explanation = self._generate_explanation(
                raw_features, features_df, model_predictions, final_prediction
            )
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'model_predictions': model_predictions,
                'model_confidences': model_confidences,
                'action_scores': action_scores,
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'model_predictions': {},
                'explanation': f'Error: {str(e)}'
            }
    
    def _generate_explanation(self, raw_features: Dict, features_df: pd.DataFrame, 
                            model_predictions: Dict, final_prediction: str) -> str:
        """Genera explicación interpretable de la predicción"""
        explanations = []
        
        # Factores principales
        if raw_features.get('rsi', 50) > 70:
            explanations.append("RSI alto indica sobrecompra")
        elif raw_features.get('rsi', 50) < 30:
            explanations.append("RSI bajo indica sobreventa")
        
        if raw_features.get('momentum_5d', 0) > 0.02:
            explanations.append("Momentum positivo fuerte")
        elif raw_features.get('momentum_5d', 0) < -0.02:
            explanations.append("Momentum negativo fuerte")
        
        # Consenso de agentes
        if raw_features.get('votes_buy', 0) >= 3:
            explanations.append("Mayoría de agentes recomiendan compra")
        elif raw_features.get('votes_sell', 0) >= 3:
            explanations.append("Mayoría de agentes recomiendan venta")
        
        # Acuerdo entre modelos
        predictions_list = list(model_predictions.values())
        if predictions_list.count(final_prediction) >= len(predictions_list) * 0.7:
            explanations.append("Alto consenso entre modelos ML")
        
        return "; ".join(explanations) if explanations else "Basado en análisis conjunto de features"
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evalúa performance de todos los modelos"""
        if not self.models:
            return {}
        
        results = {}
        
        # Preparar datos
        if hasattr(self, 'selected_features'):
            X_test = X_test[self.selected_features]
        
        X_test_scaled = self.scalers['robust'].transform(X_test)
        
        print("\n📊 EVALUACIÓN DE MODELOS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()}:")
            
            # Predicciones
            y_pred = model.predict(X_test_scaled)
            
            # Métricas
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Mostrar resultados
            for action in ['BUY', 'SELL', 'HOLD']:
                if action.lower() in report:
                    metrics = report[action.lower()]
                    print(f"  {action}: Precision={metrics['precision']:.2f}, "
                          f"Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
            
            print(f"  Accuracy: {report['accuracy']:.2f}")
            
            # ROC-AUC para clases
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    # Calcular AUC para cada clase
                    auc_scores = {}
                    for i, action in enumerate(['BUY', 'SELL', 'HOLD']):
                        if i < y_pred_proba.shape[1]:
                            y_binary = (y_test == action).astype(int)
                            auc = roc_auc_score(y_binary, y_pred_proba[:, i])
                            auc_scores[action] = auc
                    
                    avg_auc = np.mean(list(auc_scores.values()))
                    print(f"  Average AUC: {avg_auc:.3f}")
                    report['auc_scores'] = auc_scores
                except:
                    pass
            
            results[model_name] = report
        
        return results
    
    def save_models(self, path: str = 'ml_models/'):
        """Guarda modelos entrenados"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelos
        for model_name, model in self.models.items():
            joblib.dump(model, f'{path}{model_name}_model.pkl')
        
        # Guardar scalers
        joblib.dump(self.scalers, f'{path}scalers.pkl')
        
        # Guardar configuración
        config = {
            'selected_features': self.selected_features if hasattr(self, 'selected_features') else [],
            'performance_metrics': self.performance_metrics,
            'feature_importance': {k: v.to_dict() for k, v in self.feature_importance.items()},
            'model_configs': {k: v['weight'] for k, v in self.model_configs.items()},
            'trained_date': datetime.now().isoformat()
        }
        
        with open(f'{path}ml_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Modelos guardados en {path}")
    
    def load_models(self, path: str = 'ml_models/'):
        """Carga modelos pre-entrenados"""
        try:
            # Cargar modelos
            for model_name in self.model_configs.keys():
                model_path = f'{path}{model_name}_model.pkl'
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
            
            # Cargar scalers
            self.scalers = joblib.load(f'{path}scalers.pkl')
            
            # Cargar configuración
            with open(f'{path}ml_config.json', 'r') as f:
                config = json.load(f)
                self.selected_features = config['selected_features']
                self.performance_metrics = config['performance_metrics']
            
            print(f"✅ Modelos cargados desde {path}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            return False
    
    def retrain_online(self, new_data: List[Dict]):
        """Reentrenamiento online con nuevos datos"""
        if len(new_data) < 10:
            return
        
        print("\n🔄 Reentrenamiento online...")
        
        # Preparar nuevos datos
        X_new = []
        y_new = []
        
        for record in new_data:
            features = self.engineer_features(record['features'])
            X_new.append(features)
            y_new.append(record['label'])
        
        X_new = pd.concat(X_new, ignore_index=True)
        y_new = pd.Series(y_new)
        
        # Actualizar modelos (partial_fit si está disponible)
        for model_name, model in self.models.items():
            if hasattr(model, 'partial_fit'):
                if hasattr(self, 'selected_features'):
                    X_new_selected = X_new[self.selected_features]
                else:
                    X_new_selected = X_new
                
                X_new_scaled = self.scalers['robust'].transform(X_new_selected)
                model.partial_fit(X_new_scaled, y_new)
                print(f"  ✅ {model_name} actualizado")
        
        print("✅ Reentrenamiento completado")