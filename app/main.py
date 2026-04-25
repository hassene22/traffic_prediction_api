# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from datetime import datetime
import logging

from app.schemas import TrafficRequest, TrafficResponse, BatchRequest, HealthResponse
from app.model import predictor

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="Traffic Prediction API",
    description="API de prédiction de trafic en temps réel avec ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage des prédictions (pour analytics)
prediction_history = []

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=TrafficResponse)
async def predict(request: TrafficRequest, background_tasks: BackgroundTasks):
    """
    Prédiction simple du volume de trafic
    """
    try:
        # Préparation des features
        features = [
            request.hour,
            request.day_of_week,
            request.is_weekend,
            request.is_rush_hour,
            request.temperature,
            request.precipitation,
            request.historical_traffic
        ]
        
        # Prédiction
        prediction, confidence = predictor.predict(features)
        
        # Logging en background
        background_tasks.add_task(
            log_prediction, 
            features, 
            prediction, 
            confidence
        )
        
        return TrafficResponse(
            predicted_volume=round(prediction, 2),
            confidence_interval=confidence,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[TrafficResponse])
async def predict_batch(batch: BatchRequest):
    """
    Prédiction batch pour plusieurs scénarios
    """
    responses = []
    for req in batch.requests:
        features = [
            req.hour, req.day_of_week, req.is_weekend,
            req.is_rush_hour, req.temperature, 
            req.precipitation, req.historical_traffic
        ]
        prediction, confidence = predictor.predict(features)
        
        responses.append(TrafficResponse(
            predicted_volume=round(prediction, 2),
            confidence_interval=confidence,
            timestamp=datetime.now()
        ))
    
    return responses

@app.get("/stats/volume/{hour}")
async def get_hourly_average(hour: float):
    """
    Statistiques pour une heure donnée
    """
    try:
        # Simulation de stats (à remplacer par vraie base de données)
        avg_volume = 50 + 30 * np.sin(hour * 2 * np.pi / 24)
        return {
            "hour": hour,
            "average_volume": round(avg_volume, 2),
            "peak_factor": round(avg_volume / 80, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def log_prediction(features, prediction, confidence):
    """Fonction utilitaire pour logger les prédictions"""
    prediction_history.append({
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "confidence": confidence
    })
    
    # Garder seulement les 1000 dernières prédictions
    if len(prediction_history) > 1000:
        prediction_history.pop(0)
    
    logger.info(f"Prédiction effectuée: {prediction:.2f} véhicules")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )