"""
Sistema de almacenamiento y gestión de modelos ML
"""

import os
import pickle
import joblib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class ModelStorage:
    """
    Gestor de almacenamiento local de modelos ML
    """
    
    def __init__(self, base_path: str = "ml_models"):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models"
        self.backups_dir = self.base_path / "backups"
        
        # Crear directorios si no existen
        self._create_directory_structure()
        
        # Registro de modelos
        self.registry_file = self.base_path / "model_registry.json"
        self.registry = self._load_registry()
    
    def _create_directory_structure(self):
        """Crea la estructura de directorios"""
        for directory in [self.models_dir, self.backups_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self) -> Dict:
        """Carga el registro de modelos"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "active_model": None}
    
    def _save_registry(self):
        """Guarda el registro de modelos"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def save_model(self, 
                   model_name: str,
                   models: Dict[str, Any],
                   scaler: Any = None,
                   metrics: Dict[str, float] = None) -> str:
        """
        Guarda un modelo con metadata
        
        Returns:
            model_id: ID único del modelo guardado
        """
        # Generar ID único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        
        # Crear directorio para el modelo
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Guardar modelos
        models_file = model_dir / "models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(models, f)
        
        # Guardar scaler si existe
        if scaler is not None:
            scaler_file = model_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_file)
        
        # Guardar metadata
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "has_scaler": scaler is not None
        }
        
        meta_file = model_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Actualizar registro
        self.registry["models"][model_id] = metadata
        self._save_registry()
        
        return model_id
    
    def load_model(self, model_id: Optional[str] = None) -> Tuple[Dict, Any, Dict]:
        """
        Carga un modelo guardado
        
        Returns:
            Tupla (models, scaler, metadata)
        """
        # Si no se especifica, cargar el modelo activo
        if model_id is None:
            model_id = self.registry.get("active_model")
            if not model_id:
                # Cargar el más reciente
                if self.registry["models"]:
                    model_id = sorted(self.registry["models"].keys())[-1]
        
        if not model_id or model_id not in self.registry["models"]:
            raise ValueError(f"Modelo {model_id} no encontrado")
        
        # Cargar desde directorio
        model_dir = self.models_dir / model_id
        
        # Cargar modelos
        models_file = model_dir / "models.pkl"
        with open(models_file, 'rb') as f:
            models = pickle.load(f)
        
        # Cargar scaler si existe
        scaler = None
        scaler_file = model_dir / "scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
        
        # Cargar metadata
        meta_file = model_dir / "metadata.json"
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        return models, scaler, metadata
    
    def set_active_model(self, model_id: str):
        """Establece un modelo como activo"""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Modelo {model_id} no encontrado")
        
        self.registry["active_model"] = model_id
        self._save_registry()
    
    def list_models(self) -> List[Dict]:
        """Lista todos los modelos disponibles"""
        return [
            {
                "model_id": mid,
                **info
            }
            for mid, info in self.registry["models"].items()
        ]
    
    def delete_model(self, model_id: str):
        """Elimina un modelo (lo mueve a backup)"""
        if model_id not in self.registry["models"]:
            raise ValueError(f"Modelo {model_id} no encontrado")
        
        # Mover a backups
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            backup_path = self.backups_dir / model_id
            shutil.move(str(model_dir), str(backup_path))
        
        # Eliminar del registro
        del self.registry["models"][model_id]
        
        # Si era el modelo activo, limpiar
        if self.registry.get("active_model") == model_id:
            self.registry["active_model"] = None
        
        self._save_registry()

# Funciones helper para compatibilidad
def save_ml_models(ml_system, model_name: str = "trading_ml") -> str:
    """Helper para guardar modelos desde MLPredictionSystem"""
    storage = ModelStorage()
    
    models = getattr(ml_system, 'models', {})
    scaler = getattr(ml_system, 'scaler', None)
    
    metrics = {}
    if hasattr(ml_system, 'metrics'):
        metrics = ml_system.metrics
    
    model_id = storage.save_model(
        model_name=model_name,
        models=models,
        scaler=scaler,
        metrics=metrics
    )
    
    storage.set_active_model(model_id)
    return model_id

def load_ml_models(ml_system, model_id: Optional[str] = None):
    """Helper para cargar modelos en MLPredictionSystem"""
    storage = ModelStorage()
    
    models, scaler, metadata = storage.load_model(model_id)
    
    if models:
        ml_system.models = models
    if scaler is not None:
        ml_system.scaler = scaler
    
    print(f"✅ Modelos cargados: {metadata['model_id']}")
