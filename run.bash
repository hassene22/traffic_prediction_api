# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Générer les données
python data/generate_traffic_data.py

# 3. Entraîner le modèle
python notebooks/train_model.py

# 4. Lancer l'API
python app/main.py

# 5. Tester
python client_test.py