# app/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TrafficRequest(BaseModel):
    hour: float = Field(..., ge=0, le=23.99, description="Heure (0-23.99)")
    day_of_week: int = Field(..., ge=0, le=6, description="Jour de semaine (0=lundi)")
    is_weekend: int = Field(..., ge=0, le=1)
    is_rush_hour: int = Field(..., ge=0, le=1)
    temperature: float = Field(..., ge=-10, le=50)
    precipitation: float = Field(..., ge=0, le=20)
    historical_traffic: float = Field(..., ge=0, description="Volume historique (10 min)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hour": 8.5,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_rush_hour": 1,
                "temperature": 22.5,
                "precipitation": 0.0,
                "historical_traffic": 85
            }
        }

class BatchRequest(BaseModel):
    requests: list[TrafficRequest]

class TrafficResponse(BaseModel):
    predicted_volume: float
    confidence_interval: dict
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str