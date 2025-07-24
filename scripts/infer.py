# scripts/infer.py

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def load_data():
    print("Loading processed data...")
    df = pd.read_parquet(DATA_DIR / "office_data.parquet")

    features = ['square_feet', 'year_built', 'air_temperature', 'dew_temperature', 'hour', 'weekday']
    X = df[features].dropna()
    original_index = X.index
    return X, original_index, df

def load_model():
    print("Loading trained model...")
    model = joblib.load(MODEL_DIR / "ridge_model.pkl")
    return model

def run_inference():
    X, index, full_df = load_data()
    model = load_model()

    print("Running prediction...")
    y_pred = model.predict(X)

    # 음수 예측값 제거 (0으로 클리핑)
    y_pred = np.maximum(y_pred, 0)

    # 결과 결합
    result_df = full_df.loc[index].copy()
    result_df['predicted_meter_reading'] = y_pred

    # 저장
    output_path = DATA_DIR / "prediction_results.parquet"
    result_df.to_parquet(output_path, index=False)
    print(f"Saved prediction results to {output_path}")

    # 일부 확인 출력
    print(result_df[['timestamp', 'building_id', 'meter_reading', 'predicted_meter_reading']].head(10))

if __name__ == "__main__":
    run_inference()
