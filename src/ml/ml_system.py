"""
Sistema ML con scaler robusto mejorado
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.strategies.ml_prediction_system import MLPredictionSystem as MLPredictionSystemBase

class MLPredictionSystem(MLPredictionSystemBase):
    """
    Sistema ML mejorado con scaler robusto
    """
    
    def __init__(self):
        # Llamar al constructor original
        super().__init__()
        
        # Cargar modelos y scaler
        self._auto_load_models()
    
    def _auto_load_models(self):
        """Carga modelos y el scaler"""
        models_dir = Path('ml_models')
        
        # Cargar modelos
        models_file = models_dir / 'trained_models.pkl'
        if models_file.exists():
            try:
                with open(models_file, 'rb') as f:
                    self.models = pickle.load(f)
                print(f"✅ Cargados {len(self.models)} modelos")
            except Exception as e:
                print(f"⚠️ Error cargando modelos: {e}")
        
        # Intentar cargar scaler_correct.pkl primero
        scaler_file = models_dir / 'scaler_correct.pkl'
        if scaler_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
                print("✅ Scaler cargado correctamente")
                return
            except Exception as e:
                print(f"⚠️ Error cargando scaler_correct.pkl: {e}")
        
        # Si no existe, intentar con scalers.pkl
        scaler_file = models_dir / 'scalers.pkl'
        if scaler_file.exists():
            try:
                scaler_data = joblib.load(scaler_file)
                if isinstance(scaler_data, dict) and 'robust' in scaler_data:
                    self.scaler = scaler_data['robust']
                    print("✅ Scaler cargado desde scalers.pkl")
                else:
                    self.scaler = None
                    print("⚠️ Formato de scaler no reconocido")
            except Exception as e:
                print(f"⚠️ Error cargando scalers.pkl: {e}")
                self.scaler = None
        else:
            print("ℹ️ Scaler no encontrado, usando predicciones sin escalar")
            self.scaler = None
    
    def predict(self, X):
        """Predicción con manejo robusto de errores"""
        if not self.models:
            return {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'model_predictions': {},
                'explanation': 'Modelos no cargados'
            }
        
        try:
            # Asegurar que X es DataFrame
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            
            # Intentar aplicar scaler
            scaling_used = False
            X_for_prediction = X_df
            
            if self.scaler is not None:
                try:
                    # Verificar que las columnas coincidan
                    if hasattr(self.scaler, 'feature_names_in_'):
                        # Reordenar columnas para coincidir con el scaler
                        scaler_features = list(self.scaler.feature_names_in_)
                        missing_features = [f for f in scaler_features if f not in X_df.columns]
                        
                        if missing_features:
                            print(f"⚠️ Faltan características para el scaler: {missing_features}")
                        else:
                            # Seleccionar y ordenar columnas
                            X_ordered = X_df[scaler_features]
                            X_scaled = self.scaler.transform(X_ordered)
                            X_for_prediction = pd.DataFrame(X_scaled, columns=scaler_features, index=X_df.index)
                            scaling_used = True
                    else:
                        # Intentar escalar directamente
                        X_scaled = self.scaler.transform(X_df)
                        X_for_prediction = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)
                        scaling_used = True
                        
                except Exception as e:
                    print(f"⚠️ No se pudo aplicar scaler: {str(e)[:50]}")
                    # Continuar sin scaler
            
            # Hacer predicciones
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_for_prediction)
                    predictions[name] = int(pred[0])
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_for_prediction)
                        probabilities[name] = proba[0]
                        
                except Exception as e:
                    print(f"⚠️ Error en {name}: {str(e)[:30]}")
                    continue
            
            if not predictions:
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.5,
                    'model_predictions': {},
                    'explanation': 'No se pudieron hacer predicciones'
                }
            
            # Calcular predicción ensemble
            pred_values = list(predictions.values())
            pred_counts = pd.Series(pred_values).value_counts()
            
            final_pred = pred_counts.index[0]
            confidence = pred_counts.iloc[0] / len(pred_values)
            
            # Mejorar confianza con probabilidades
            if probabilities:
                conf_scores = []
                for name, proba in probabilities.items():
                    if predictions[name] == final_pred and len(proba) > final_pred:
                        conf_scores.append(proba[final_pred])
                if conf_scores:
                    confidence = np.mean(conf_scores)
            
            # Mapear predicciones
            pred_map = {0: 'SELL', 1: 'BUY'}
            
            # Ajustar si está cerca de 50%
            if 0.48 < confidence < 0.52:
                final_prediction = 'HOLD'
            else:
                final_prediction = pred_map.get(final_pred, 'HOLD')
            
            return {
                'prediction': final_prediction,
                'confidence': float(confidence),
                'model_predictions': {name: pred_map.get(pred, 'HOLD') 
                                    for name, pred in predictions.items()},
                'explanation': f'Predicción {"con" if scaling_used else "sin"} scaler',
                'scaled': scaling_used
            }
            
        except Exception as e:
            return {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'model_predictions': {},
                'explanation': f'Error: {str(e)[:100]}'
            }
