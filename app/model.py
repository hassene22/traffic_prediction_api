# app/model.py
import joblib
import numpy as np
from pathlib import Path

class TrafficPredictor:
    def __init__(self, model_path: str = "models/traffic_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"✅ Modèle chargé depuis {self.model_path}")
        else:
            print(f"⚠️ Modèle non trouvé à {self.model_path}")
            self.model = None
    
    def predict(self, features: list) -> float:
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        
        # Intervalle de confiance simplifié
        std_dev = 15  # Estimé à partir des erreurs du modèle
        confidence = {
            "lower": max(0, prediction - 1.96 * std_dev),
            "upper": prediction + 1.96 * std_dev
        }
        
        return prediction, confidence

# Instance globale
predictor = TrafficPredictor()