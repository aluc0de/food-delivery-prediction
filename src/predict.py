import pandas as pd
import joblib
import sys
from dataclasses import dataclass, asdict

RESET = "\033[0m"

#estrutura do pedido 
@dataclass
class DeliveryOrder:
    Order_Hour: int
    Is_Weekend: int
    Is_Festival: int
    Rider_Experience_Years: float
    Rider_Rating: float
    Restaurant_Rating: float
    Order_Items: int
    Preparation_Time_Min: int
    Road_Distance_km: float
    Number_of_Signals: int
    Weather: str
    Traffic_Level: str
    Vehicle_Type: str
    Cuisine_Type: str
    Restaurant_Load: str
    Delivery_Priority: str


#recebe o objeto DeliveryOrder e retorna a previsão em tempo
def predict_delivery_time(order: DeliveryOrder):
    model = joblib.load('../models/random_forest_model.pkl')
    model_columns = joblib.load('../models/model_columns.pkl')
    
    #transforma em um dataframe, revisa e alinha as colunas pro RandomForest não travar
    order_dict = asdict(order)
    
    df_new = pd.DataFrame([order_dict])
    df_new_encoded = pd.get_dummies(df_new)
    df_new_encoded = df_new_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_new_encoded)[0]
    return prediction

#easter egg
def show_batcave():
    yellow = "\033[1;33m"
    white = "\033[1;37m"
    grey = "\033[90m"
    print(f"\n{grey}==============================================={RESET}")
    print(f"{yellow}        I AM VENGEANCE. I AM THE NIGHT.        {RESET}")
    print(f"{white}         I AM ALUC0DE THE DEVELOPER.          {RESET}")
    print(f"{grey}==============================================={RESET}\n")


#bloco principal
if __name__ == "__main__":
    #verifica se foi passado argumento no terminal para o easter egg rodar
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "batcave":
            show_batcave()
        else:
            print(f"Unknown command: '{command}'. Try running without arguments.")

    else:
        example_order = DeliveryOrder(
            Order_Hour=19,
            Is_Weekend=0,
            Is_Festival=0,
            Rider_Experience_Years=3.0,
            Rider_Rating=4.5,
            Restaurant_Rating=4.2,
            Order_Items=2,
            Preparation_Time_Min=15,
            Road_Distance_km=8.5,
            Number_of_Signals=12,
            Weather='Rain',
            Traffic_Level='High',
            Vehicle_Type='Scooter',
            Cuisine_Type='Pizza',
            Restaurant_Load='High',
            Delivery_Priority='Normal'
        )

        estimated_time = predict_delivery_time(example_order)
        print(f"Estimated delivery time: {estimated_time:.1f} minutes")
