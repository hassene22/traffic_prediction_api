# client_test.py
import requests
import time

API_URL = "http://localhost:8000"

def test_single_prediction():
    """Test une prédiction simple"""
    data = {
        "hour": 8.5,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "temperature": 22.5,
        "precipitation": 0.0,
        "historical_traffic": 85
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print("📊 Prédiction simple:", response.json())

def test_load():
    """Test de charge simple"""
    start = time.time()
    for i in range(100):
        test_single_prediction()
    end = time.time()
    print(f"⚡ 100 prédictions en {end-start:.2f} secondes")

if __name__ == "__main__":
    test_single_prediction()
    test_load()