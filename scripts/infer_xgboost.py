import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import argparse

# 경로 설정
DATA_PATH = Path("data/processed/office_data_merged.parquet")
MODEL_DIR = Path("models")
RESULT_DIR = Path("data/processed")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

def infer_xgboost(meter_type_input: str):
    print(f"Inference for meter_type = '{meter_type_input}'")

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

    df = df.dropna(subset=features)
    print(f"Remaining after dropna: {df.shape}")

    if df.empty:
        raise ValueError("No data available for inference.")

    X = df[features]

    # canonical name for loading model and saving result
    name_map = {v: k for k, v in meter_type_mapping.items() if not k.isdigit()}
    canonical_name = name_map[raw_code]
    model_path = MODEL_DIR / f"xgboost_model_{canonical_name}.json"

    print(f"Loading model from {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    print("Running inference...")
    predictions = model.predict(X)

    df_result = df.copy()
    df_result["prediction"] = predictions

    output_path = RESULT_DIR / f"inference_results_{canonical_name}.parquet"
    df_result.to_parquet(output_path, index=False)
    print(f"Saved inference results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter_type", type=str, required=True, help="Meter type name or code (e.g., 'electricity' or '0')")
    args = parser.parse_args()

    infer_xgboost(args.meter_type)
