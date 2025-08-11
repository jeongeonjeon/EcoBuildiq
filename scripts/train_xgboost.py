import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import argparse

# 경로 설정
DATA_PATH = Path("data/processed/office_data_merged.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# meter_type 매핑
meter_type_mapping = {
    "electricity": 0, "0": 0,
    "chilledwater": 1, "1": 1,
    "steam": 2, "2": 2,
    "hotwater": 3, "3": 3,
    "gas": 4, "4": 4,
    "water": 5, "5": 5,
    "irrigation": 6, "6": 6,
    "solar": 7, "7": 7,
}

def train_xgboost(meter_type_input: str):
    print(f"Training for meter_type = '{meter_type_input}'")

    if meter_type_input not in meter_type_mapping:
        raise ValueError(f"Invalid meter_type input: {meter_type_input}")

    raw_code = meter_type_mapping[meter_type_input]
    df = pd.read_parquet(DATA_PATH)

    # 정규화된 meter_type 값 중 가장 가까운 값을 선택
    closest_val = min(df["meter_type"].unique(), key=lambda x: abs(x - raw_code))
    df = df[np.isclose(df["meter_type"], closest_val, atol=1e-4)]
    print(f"Filtered data shape: {df.shape}")

    features = [
        "meter_type", "hour", "dayofweek", "is_weekend", "holiday",
        "air_temperature", "dew_temperature", "wind_speed",
        "cloud_coverage", "precip_depth_1_hr", "square_feet", "year_built"
    ]
    target = "value"

    df = df.dropna(subset=features + [target])
    print(f"Remaining after dropna: {df.shape}")

    if df.empty:
        raise ValueError("No training data available after filtering. Please check meter_type or data quality.")

    X = df[features]
    y = df[target]

    print("Splitting dataset")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Evaluating model")
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE : {mae:.4f}")

    # canonical name for saving model
    name_map = {v: k for k, v in meter_type_mapping.items() if not k.isdigit()}
    canonical_name = name_map[raw_code]
    model_path = MODEL_DIR / f"xgboost_model_{canonical_name}.json"

    print(f"Saving model to {model_path}")
    model.save_model(str(model_path))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter_type", type=str, required=True, help="Meter type name or code (e.g., 'electricity' or '0')")
    args = parser.parse_args()

    train_xgboost(args.meter_type)
