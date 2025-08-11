import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

# 경로 설정
INPUT_PATH = Path("data/processed/merged_energy_weather.parquet")
OUTPUT_PATH = Path("data/processed/office_data_merged.parquet")

# 1. 병합된 에너지+날씨 데이터 로드
df = pd.read_parquet(INPUT_PATH)
print("Original shape:", df.shape)

# 2. 필수 열 존재 확인 및 정렬
df = df.dropna(subset=["timestamp", "building_id", "value", "meter_type"])
df = df.sort_values("timestamp")

# 3. 시간 파생 변수 생성
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["holiday"] = df["timestamp"].dt.strftime('%Y-%m-%d').isin([
    # 여기에 공휴일 리스트를 입력할 수 있음
]).astype(int)

# meter_type을 숫자 코드로 변환하여 정규화 가능하도록 처리
meter_type_mapping = {
    "electricity": 0,
    "chilledwater": 1,
    "steam": 2,
    "hotwater": 3,
    "gas": 4,
    "water": 5,
    "irrigation": 6,
    "solar": 7
}
df["meter_type"] = df["meter_type"].map(meter_type_mapping)

# 4. 사용할 기본 feature 정의 (타겟은 value)
candidate_features = [
    "meter_type", "hour", "dayofweek", "is_weekend", "holiday",
    "air_temperature", "dew_temperature", "wind_speed", "cloud_coverage", "precip_depth_1_hr",
    "square_feet", "year_built", "floor_count", "indoor_temperature", "humidity", "occupancy"
]

# 5. 실제 존재하며 결측 비율이 10% 이하인 feature만 선택
feature_columns = [
    col for col in candidate_features if col in df.columns and df[col].notna().sum() > 0.9 * len(df)
]

print("Number of features used:", len(feature_columns))
print("Used features:", feature_columns)

# 6. 정규화 (결측값 있는 열은 제외하고 정규화 수행)
scaler = StandardScaler()
df_scaled = df.copy()
scalable_cols = [col for col in feature_columns if df_scaled[col].notna().all()]
df_scaled[scalable_cols] = scaler.fit_transform(df_scaled[scalable_cols])

# 7. 결과 저장
os.makedirs("data/processed", exist_ok=True)
df_scaled.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")
