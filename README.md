# FOOD DELIVERY TIME PREDICTION

## DESCRIPTION

If you're reading this, you probably just stumbled upon this repository. Either way, welcome!

This project is a Machine Learning model designed to predict food delivery time based on order characteristics, traffic, and weather conditions. I built this from scratch as part of my journey through the **Associate AI Engineer for Developers** Career Track on DataCamp. My goal was to apply the concepts I learned in a real-world-like scenario, and to think like an engineer when dealing with data.

## FEATURES

- Exploratory Data Analysis (EDA) to understand the distribution of delivery times.
- Data cleaning and One-Hot Encoding for categorical variables.
- Training of a Linear Regression baseline and a Random Forest Regressor.
- Evaluation using MAE, RMSE, and R².
- Feature importance analysis to understand what drives delivery time.
- A modular codebase with separate files for preprocessing, training, and prediction.

## DESIGN DECISIONS AND CHALLENGES

**The Synthetic Data Trap:** The dataset is synthetic, which means the relationships between features (like distance and time) were artificially generated. I had to be honest about this. A model could easily achieve a 0.98 R² by memorizing the rules. Instead of claiming real-world performance, I focused on the methodology and documented this as a limitation.

**The Data Leakage Scare:** During EDA, I noticed that `Average_Speed_kmph` had a high correlation with the target. But wait—the average speed is only known after the delivery is done. If I fed this to the model, it would be cheating. I removed it. The model must learn from what is known *before* the delivery starts.

**The One-Hot Encoding Explosion:** Variables like Weather and Vehicle Type needed to become numbers. The `pd.get_dummies` function transformed 19 features into 46 columns. To avoid errors when making predictions, I saved the model columns to a `.pkl` file and used `reindex` to align the new data.

**The Smart App Control Blockade:** Windows 11's Smart App Control blocked DLL files from Pandas and Scikit-learn. I had to disable it to run the environment. It was a real-world reminder that engineering is often about fighting the environment as much as writing code.

## PROJECT FILES

- `notebooks/01_data_exploration.ipynb`: The Jupyter Notebook with all the EDA, model comparison, and sensitivity analysis.
- `src/preprocessing.py`: The function to clean the data and apply One-Hot Encoding.
- `src/train.py`: The script to train the Random Forest and save the model.
- `src/predict.py`: The script to load the model and predict the delivery time for a new order.
- `models/`: The folder where the trained model and column list are saved. (generated after running train.py; not versioned in this repository)

## REQUIREMENTS AND USAGE

This project uses Python and requires the following libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, and joblib. You can install them by running `pip install -r requirements.txt`.

Before running the scripts, you need to download the dataset from Kaggle and place it in the `data/` folder. 

To train the model, go to the `src` folder and run `python train.py`. 

To make a prediction with a sample order, run `python predict.py`.

And if you are curious enough, maybe you can find a hidden command...

## FINAL THOUGHTS

There's still room to improve this project. If I had real data, I would use a time-based split (past data to predict future) instead of a random split. I would also test with real delivery data to see if the model's logic holds up in the real world.

I'm proud of how far this project has come, and I'm looking forward to applying what I've learned to my next projects.

Thank you for taking the time to look at my work!

This was me, Goodbye!
