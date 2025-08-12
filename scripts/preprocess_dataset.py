import pandas as pd
import numpy as np
import os
from pathlib import Path

# 경로 설정
INPUT_PATH = Path("data/processed/merged_energy_weather.parquet")
OUTPUT_PATH = Path("data/processed/office_data_merged.parquet")

# 1) 필요한 컬럼만 읽기 (열 프로젝션으로 메모리 절감)
#    merge 단계에서 보장되는 표준 컬럼 + 날씨만
use_cols = [
    "timestamp", "building_id", "meter_type", "value", "source",
    "site_id", "square_feet", "year_built", "floor_count",
    "air_temperature", "dew_temperature", "wind_speed",
    "cloud_coverage", "precip_depth_1_hr",
    # 있으면 읽힘, 없으면 무시됨
    "sea_level_pressure", "wind_direction"
]

print("Reading parquet (projected columns only)...")
df = pd.read_parquet(INPUT_PATH, columns=[c for c in use_cols if c in pd.read_parquet(INPUT_PATH, columns=None).columns])
print("Original (projected) shape:", df.shape)

# 2) 필수 결측 제거 + 시간 정렬 (inplace로 복제 방지)
df.dropna(subset=["timestamp", "building_id", "value", "meter_type"], inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df.sort_values("timestamp", inplace=True)

# 3) 시간 파생 (벡터 연산, inplace)
hour = df["timestamp"].dt.hour
dow  = df["timestamp"].dt.dayofweek
df["is_weekend"] = (dow >= 5).astype("int8")
df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
df["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0).astype("float32")
df["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0).astype("float32")
# (옵션) holiday는 뺍니다. 리스트가 길면 메모리/시간 부담 ↑
# 필요하면 나중에 소수의 특정 날짜만 체크해서 int8로 넣으세요.

# 4) 타입 다운캐스트 (메모리 절감)
# 숫자형
for col in ["value", "air_temperature", "dew_temperature", "wind_speed",
            "cloud_coverage", "precip_depth_1_hr", "sea_level_pressure", "wind_direction",
            "square_feet"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

if "year_built" in df.columns:
    df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce").astype("float32")
if "floor_count" in df.columns:
    df["floor_count"] = pd.to_numeric(df["floor_count"], errors="coerce").astype("float32")

# 범주형(문자열) -> category (메모리 절감)
for col in ["building_id", "site_id", "source", "meter_type"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# 5) meter_type은 여기서 숫자/원-핫으로 바꾸지 않습니다.
#    → lstm_prepare.py에서 필요 시 원-핫/매핑 처리하세요.

# 6) 저장 (압축 스니피, row_group은 fastparquet 기본)
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}  | shape: {df.shape}  | dtypes:")
print(df.dtypes)
