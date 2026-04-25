# data/generate_traffic_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Générer 10,000 échantillons
n_samples = 10000
dates = [datetime(2024, 1, 1) + timedelta(minutes=15*i) for i in range(n_samples)]

# Caractéristiques
data = {
    'timestamp': dates,
    'hour': [d.hour + d.minute/60 for d in dates],
    'day_of_week': [d.weekday() for d in dates],
    'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
    'is_rush_hour': [1 if (7 <= d.hour <= 9) or (17 <= d.hour <= 19) else 0 for d in dates],
    'temperature': np.random.uniform(0, 40, n_samples),
    'precipitation': np.random.exponential(0.5, n_samples),
    'historical_traffic': np.random.poisson(100, n_samples),
}

# Target: volume de trafic (voitures/heure)
# Formule plus réaliste avec interactions
traffic_volume = (
    50 +  # base
    20 * np.sin(data['hour'] * 2 * np.pi / 24) +  # cycle journalier
    30 * np.sin(data['hour'] * 2 * np.pi / 12) +   # pic matin/soir
    15 * data['is_rush_hour'] +  # effet rush hour
    -10 * data['is_weekend'] +   # weekend moins chargé
    5 * np.sin(data['day_of_week'] * 2 * np.pi / 7) +  # cycle hebdomadaire
    -3 * (data['temperature'] - 20)**2 / 100 +  # température optimale 20°C
    -20 * data['precipitation'] +  # pluie réduit trafic
    0.7 * data['historical_traffic'] +  # effet mémoire
    np.random.normal(0, 15, n_samples)  # bruit
)

traffic_volume = np.maximum(traffic_volume, 10)  # minimum 10 voitures
data['traffic_volume'] = traffic_volume

df = pd.DataFrame(data)
df.to_csv('data/traffic_data.csv', index=False)
print(f"✅ Généré {n_samples} échantillons")