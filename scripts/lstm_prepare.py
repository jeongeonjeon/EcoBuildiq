"""
- 사용 데이터: office_data_merged.parquet
- 출력 파일: X_lstm.npy, y_lstm.npy ([value, meter_type])
- 시각화 포함
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import sys
import os
from tqdm import tqdm

# # print(os.getcwd())

# script_path = os.path.abspath("../scripts")
# if script_path not in sys.path:
#     sys.path.append(script_path)

# print(script_path)
from utils import (
    meter_name_to_id,
    meter_id_to_name,
    decode_meter_type,
    inverse_transform
)

# 설정
INPUT_PATH = "data/processed/office_data_merged.parquet"
OUTPUT_DIR = Path("data/processed")
SCALER_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)
# X_OUTPUT = os.path.join(OUTPUT_DIR, "X_lstm.npy")
# Y_OUTPUT = os.path.join(OUTPUT_DIR, "y_lstm.npy")
# SCALER_OUTPUT = Path("models/standard_scaler.pkl")

# # meter_type 디코딩
# df["meter_type"] = df["meter_type"].apply(decode_meter_type)

# 조건 필터링 (예시)
# df_all = df_all[
    # (df["meter_type"] == "electricity") 
    # (df_all["source"] == "ASHRAE") 
    # & (df_all["building_id"] == "Hog_office_Cornell")
# ].copy()
# print("After filtering:", df_all.shape)

# 입력 및 타겟 변수 설정
input_cols = [
    "air_temperature", "dew_temperature", "sea_level_pressure", "wind_direction",
    "wind_speed", "cloud_coverage", "square_feet", "hour", "dayofweek",
    "is_weekend", "holiday", "meter_type", "value"
]
target_cols = ["value", "air_temperature"]  # occupancy는 나중에 추가 예정
meter_types = ["electricity","chilledwater", "steam" , "hotwater", "gas", "water", "irrigation", "solar"]
# meter_type_mapping = {
#     "electricity": 0,
#     "chilledwater": 1,
#     "steam": 2,
#     "hotwater": 3,
#     "gas": 4,
#     "water": 5,
#     "irrigation": 6,
#     "solar": 7
# }

window_size = 24
chunk_size = 100000  # 슬라이딩 윈도우 처리 시 한 번에 처리할 row 수


# 데이터 불러오기
df_all = pd.read_parquet(INPUT_PATH)
print("Original shape:", df_all.shape)


# 결측치 제거 및 정렬
# df_all = df_all.dropna(subset=input_cols + target_cols)
# df_all = df_all.sort_values("timestamp")

# 정규화용 샘플링
df_chuck = df_all.dropna(subset=input_cols + target_cols).sort_values("timestamp")
df_chuck = df_chuck.sample(n=min(chunk_size, len(df_chuck)), random_state=42)
df_chuck["meter_type"] = df_chuck["meter_type"].apply(decode_meter_type)
df_chuck["meter_type"] = df_chuck["meter_type"].map(meter_name_to_id)

print("After dropna & sort:", df_all.shape)

scaler_X = StandardScaler().fit(df_chuck[input_cols].values)
scaler_y = StandardScaler().fit(df_chuck[target_cols].values)

joblib.dump(scaler_X, SCALER_DIR / "scaler_X_all.pkl")
joblib.dump(scaler_y, SCALER_DIR / "scaler_y_all.pkl")


# meter_type별 슬라이딩 윈도우 변환 및 저장
for meter in meter_types:
    print(f"Processing meter_type: {meter}")

    df = df_all.copy()
    df["meter_type"] = df["meter_type"].apply(decode_meter_type)
    df = df[df["meter_type"] == meter].copy()

    if df.empty:
        print(f"No data for meter_type={meter}, skipping.")
        continue

    df = df.dropna(subset=input_cols + target_cols)
    df = df.sort_values("timestamp")
    df["meter_type"] = df["meter_type"].map(meter_name_to_id)

    df_X = df[input_cols].values
    df_y = df[target_cols].values

    X_path = OUTPUT_DIR / f"X_lstm_{meter}.npy"
    y_path = OUTPUT_DIR / f"y_lstm_{meter}.npy"

    n_seq_estimate = len(df) - window_size
    X_memmap = np.lib.format.open_memmap(X_path, dtype='float32', mode='w+', shape=(n_seq_estimate, window_size, len(input_cols)))
    y_memmap = np.lib.format.open_memmap(y_path, dtype='float32', mode='w+', shape=(n_seq_estimate, len(target_cols)))

    for i in tqdm(range(n_seq_estimate), desc=f"{meter} sequences"):
        try:
            X_window = df_X[i:i+window_size]
            y_target = df_y[i+window_size]

            X_scaled = scaler_X.transform(X_window).astype('float32')
            y_scaled = scaler_y.transform([y_target])[0].astype('float32')

            X_memmap[i] = X_scaled
            y_memmap[i] = y_scaled
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue

    print(f"Saved {n_seq_estimate} sequences for {meter} to {X_path.name} and {y_path.name}")



