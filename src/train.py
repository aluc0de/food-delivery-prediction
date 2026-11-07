import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocessing import clean_and_prepare_data

def train_model():

# Verifica se o dataset existe antes de tentar ler
    data_path = '../data/Food_Delivery_Time_Prediction.csv'

    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at '{data_path}'.")
        print("Please download it from Kaggle and place it in the 'data/' folder.")
        print("Kaggle link: https://www.kaggle.com/datasets/dharmendrapandit12/food-delivery-time-prediction-dataset")
        sys.exit(1) # Para o programa antes dele tentar ler o resto das coisas
        
    # 1. Carregar os dados
    print("Loading data...")
    df = pd.read_csv('../data/Food_Delivery_Time_Prediction.csv')

    # 2. Preparar os dados usando a função do preprocessing.py
    print("Preparing data...")
    X, y = clean_and_prepare_data(df)

    # 3. Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Treinar o modelo
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 5. Salvar o modelo e as colunas
    os.makedirs('../models', exist_ok=True)
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    joblib.dump(X_train.columns.tolist(), '../models/model_columns.pkl')
    
    print("Model trained and saved in the models/ folder!")

if __name__ == "__main__":
    train_model()
