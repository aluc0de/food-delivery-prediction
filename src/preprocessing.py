

import pandas as pd

def clean_and_prepare_data(df):
    """
    Recebe o DataFrame bruto e retorna as features (X) e o alvo (y).
    Remove colunas com data leakage e aplica One-Hot Encoding.
    """
    # Separar a variável alvo
    y = df['Time_taken_min']

    # Remover colunas que não devem ser usadas 
    colunas_para_remover = [
        'Time_taken_min', 
        'Order_ID', 
        'Order_Date', 
        'Average_Speed_kmph', 
        'Delivery_Distance_Category'
    ]
    X = df.drop(columns=colunas_para_remover)

    # One-Hot Encoding para variáveis de texto e tira redundâncias com o "drop first"
    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y
