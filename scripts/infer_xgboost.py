import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# 경로 설정
DATA_PATH = Path("data/processed/office_data.parquet")
MODEL_PATH = Path("models/xgboost_model.json")
OUTPUT_PATH = Path("data/processed/prediction_results_xgboost.parquet")

def run_inference():
    print("Loading processed data...")
    df = pd.read_parquet(DATA_PATH)

    # 사용할 특성과 타겟
    features = ['square_feet', 'year_built', 'air_temperature', 'dew_temperature', 'hour', 'weekday']
    df = df.dropna(subset=features)

    X = df[features]

    print("Loading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    print("Running prediction...")
    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)  # 음수 제거

    print("Saving results...")
    result_df = df[['timestamp', 'building_id']].copy()
    result_df['predicted_meter_reading'] = y_pred
    result_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved prediction results to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference()
