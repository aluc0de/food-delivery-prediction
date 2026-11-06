import pandas as pd
import joblib
import sys

RESET = "\033[0m"

def predict_delivery_time(order_dict):
    """
    Recebe um dicionário com os dados de um pedido e retorna a previsão em minutos.
    """
    # 1. Carregar o modelo treinado e as colunas esperadas
    model = joblib.load('../models/random_forest_model.pkl')
    model_columns = joblib.load('../models/model_columns.pkl')

    # 2. Transformar o pedido em um DataFrame
    df_new = pd.DataFrame([order_dict])

    # 3. Aplicar o mesmo One-Hot Encoding usado no treinamento
    df_new_encoded = pd.get_dummies(df_new)
    
    # 4. Alinhar as colunas com o que o modelo espera (preenchendo o que faltar com 0)
    df_new_encoded = df_new_encoded.reindex(columns=model_columns, fill_value=0)

    # 5. Fazer a previsão
    prediction = model.predict(df_new_encoded)[0]
    
    return prediction

def show_batcave():
    yellow = "\033[1;33m"
    white = "\033[1;37m"
    grey = "\033[90m"
    print(f"\n{grey}==============================================={RESET}")
    print(f"{yellow}        I AM VENGEANCE. I AM THE NIGHT.        {RESET}")
    print(f"{white}         I AM ALUC0DE THE DEVELOPER.          {RESET}")
    print(f"{grey}==============================================={RESET}\n")

if __name__ == "__main__":
    # Verifica se o usuário passou algum argumento no terminal
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "batcave":
            show_batcave()
        else:
            print(f"Unknown command: '{command}'. Try running without arguments.")
    
    else:
        # Se nenhum argumento for passado, roda o exemplo padrão de previsão
        example_order = {
            'Order_Hour': 19,
            'Is_Weekend': 0,
            'Is_Festival': 0,
            'Rider_Experience_Years': 3.0,
            'Rider_Rating': 4.5,
            'Restaurant_Rating': 4.2,
            'Order_Items': 2,
            'Preparation_Time_Min': 15,
            'Road_Distance_km': 8.5,
            'Number_of_Signals': 12,
            'Weather': 'Rain',
            'Traffic_Level': 'High',
            'Vehicle_Type': 'Scooter',
            'Cuisine_Type': 'Pizza',
            'Restaurant_Load': 'High',
            'Delivery_Priority': 'Normal'
        }
        
        estimated_time = predict_delivery_time(example_order)
        print(f"Estimated delivery time: {estimated_time:.1f} minutes")