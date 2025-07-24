# scripts/train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path

DATA = Path("data/processed")
MODEL = Path("models")

def train_ridge():
    print("Loading preprocessed data...")
    df = pd.read_parquet(DATA / "office_data.parquet")

    features = ['square_feet', 'year_built', 'air_temperature', 'dew_temperature', 'hour', 'weekday']
    target = 'meter_reading'

    print("Splitting dataset...")
    X = df[features].dropna()
    y = df.loc[X.index, target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Ridge Regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(((y_val - y_pred) ** 2).mean())  # rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE: {rmse:.4f}")

    print("Saving model...")
    MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL / "ridge_model.pkl")

if __name__ == "__main__":
    train_ridge()
