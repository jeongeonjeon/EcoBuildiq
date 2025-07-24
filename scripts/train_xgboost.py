import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib

# 경로 설정
DATA_PATH = Path("data/processed/office_data.parquet")
MODEL_PATH = Path("models/xgboost_model.json")

def train_xgboost():
    print("Loading processed data...")
    df = pd.read_parquet(DATA_PATH)

    # 특성 선택
    features = ['square_feet', 'year_built', 'air_temperature', 'dew_temperature', 'hour', 'weekday']
    target = 'meter_reading'

    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE : {mae:.4f}")

    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(str(MODEL_PATH))
    print("Training complete.")

if __name__ == "__main__":
    train_xgboost()
