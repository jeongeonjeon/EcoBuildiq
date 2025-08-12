"""
- 입력: data/processed/office_data_merged.parquet  (전처리에서 스케일링/원핫 안 함)
- 출력: data/processed/X_lstm_{meter}.npy, y_lstm_{meter}.npy
- 특징:
  1) 건물 경계를 절대 넘지 않도록 (source, site_id, building_id, meter_type) 그룹 내에서만 윈도우 생성
  2) 입력 X만 StandardScaler로 스케일 (y는 원 스케일 또는 log1p 스케일로 저장)
  3) STRIDE 적용으로 중복/용량 감소, X는 float16로 저장 (학습 시 torch가 float32로 캐스팅)
  4) meter_type은 one-hot으로 입력에 포함 (mt_* 컬럼)
  5) window_size=96 -> 48 -> 36로 완화 (필요 시 조정)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# 설정
# =========================
INPUT_PATH = Path("data/processed/office_data_merged.parquet")
OUTPUT_DIR = Path("data/processed")
SCALER_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)

# 시퀀스 파라미터
window_size = 36
STRIDE = 4

# 저장 dtype
DTYPE_X = "float16"
DTYPE_Y = "float32"

# 타깃 스케일 (log1p 사용 권장: 스파이크 완화)
USE_LOG1P_TARGET = True   # True면 y = log1p(value), False면 원 스케일 value

# 입력 컬럼 자동 구성 규칙
TIME_COLS   = ["is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]  # 전처리에서 생성
WEATHER_CANDIDATES = [
    "air_temperature", "dew_temperature", "wind_speed",
    "cloud_coverage", "precip_depth_1_hr", "sea_level_pressure", "wind_direction"
]
STATIC_CANDIDATES  = ["square_feet", "year_built", "floor_count"]


def _safe_cols(df: pd.DataFrame, candidates):
    return [c for c in candidates if c in df.columns]


def _count_windows(n_rows: int, win: int, stride: int) -> int:
    n_seq_est = n_rows - win
    if n_seq_est <= 0:
        return 0
    return ((n_seq_est + stride - 1) // stride)


def main():
    print(f"Loading: {INPUT_PATH}")
    df_all = pd.read_parquet(INPUT_PATH)
    # 필수 컬럼 체크
    must_cols = ["timestamp", "building_id", "meter_type", "value"]
    for c in must_cols:
        if c not in df_all.columns:
            raise ValueError(f"Missing required column: {c}")

    # 시간 정렬 & 타입 기본 정리(전처리에서 이미 돼 있어야 하지만 재확인)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["timestamp", "building_id", "meter_type", "value"]).sort_values("timestamp")

    # meter_type one-hot (입력 특징에 포함)
    # category → get_dummies(int dtype로)
    if df_all["meter_type"].dtype.name != "category":
        df_all["meter_type"] = df_all["meter_type"].astype("category")
    mt_dummies = pd.get_dummies(df_all["meter_type"], prefix="mt", dtype=np.int8)
    df_all = pd.concat([df_all, mt_dummies], axis=1)

    # 입력 컬럼 자동 구성
    time_cols   = _safe_cols(df_all, TIME_COLS)
    weather_cols= _safe_cols(df_all, WEATHER_CANDIDATES)
    static_cols = _safe_cols(df_all, STATIC_CANDIDATES)
    mt_cols     = sorted([c for c in df_all.columns if c.startswith("mt_")])

    # value / value_log 구성
    if USE_LOG1P_TARGET:
        # 음수 방지를 위해 value<0은 0으로 클리핑 후 log1p
        df_all["value_log"] = np.log1p(np.clip(df_all["value"].astype("float32"), a_min=0, a_max=None))
        y_col = "value_log"
        # 입력에도 과거값 특징을 써야 하면 value_log 사용
        value_feature_col = "value_log"
    else:
        y_col = "value"
        value_feature_col = "value"

    # 최종 입력 컬럼
    input_cols = time_cols + weather_cols + static_cols + mt_cols + [value_feature_col]
    print(f"Selected input_cols ({len(input_cols)}): {input_cols}")
    print(f"Target column: {y_col}")

    # -------------------------
    # 스케일러 피팅 (입력 X만)
    # 샘플링: 결측 없는 행에서 일부만 사용 (메모리 절약)
    # -------------------------
    need_cols_for_scale = input_cols  # y_col은 제외(원하면 포함 x)
    df_sample = df_all.dropna(subset=need_cols_for_scale + [y_col]).sample(
        n=min(100_000, len(df_all)), random_state=42
    )
    scaler_X = StandardScaler().fit(df_sample[input_cols].values.astype("float32"))
    joblib.dump(scaler_X, SCALER_DIR / "scaler_X_all.pkl")
    print("Saved scaler_X_all.pkl")

    # =========================
    # 2-패스: meter별 총 윈도우 수 계산 → memmap 할당 → 그룹별로 채우기
    # 그룹 키: (source, site_id, building_id, meter_type)
    # =========================
    # 그룹 전, 누락 키 보완
    for col in ["source", "site_id"]:
        if col not in df_all.columns:
            df_all[col] = "NA"

    group_keys = ["source", "site_id", "building_id", "meter_type"]
    grouped = df_all.groupby(group_keys, observed=True, sort=False)

    # meter 목록
    meters = list(df_all["meter_type"].cat.categories)
    print(f"Meters found: {meters}")

    # ---- 1 PASS: meter별 총 윈도우 수 계산
    total_per_meter = {m: 0 for m in meters}

    print("Counting windows per meter (pass 1)...")
    for (src, site, bid, mtype), g in tqdm(grouped, total=len(grouped)):
        g = g.dropna(subset=input_cols + [y_col]).sort_values("timestamp")
        n_out = _count_windows(len(g), window_size, STRIDE)
        if n_out > 0:
            total_per_meter[mtype] += n_out

    # ---- 2 PASS: 저장 및 채우기
    for meter in meters:
        total_out = total_per_meter.get(meter, 0)
        if total_out <= 0:
            print(f"[{meter}] no windows, skip.")
            continue

        X_path = OUTPUT_DIR / f"X_lstm_{meter}.npy"
        y_path = OUTPUT_DIR / f"y_lstm_{meter}.npy"

        # memmap 생성
        # feature 수는 input_cols 길이
        X_memmap = np.lib.format.open_memmap(
            X_path, dtype=DTYPE_X, mode="w+",
            shape=(total_out, window_size, len(input_cols))
        )
        y_memmap = np.lib.format.open_memmap(
            y_path, dtype=DTYPE_Y, mode="w+",
            shape=(total_out, 1)  # target은 단일(value_log 또는 value)
        )

        out_idx = 0
        print(f"[{meter}] writing {total_out} windows...")
        for (src, site, bid, mtype), g in grouped:
            if mtype != meter:
                continue

            g = g.dropna(subset=input_cols + [y_col]).sort_values("timestamp")
            n_out = _count_windows(len(g), window_size, STRIDE)
            if n_out <= 0:
                continue

            # numpy 배열로 뽑기 (속도/메모리 균형)
            gX = g[input_cols].values.astype("float32")
            gy = g[y_col].values.astype("float32")

            # 슬라이딩 윈도우 작성 (STRIDE 적용)
            for i in range(0, len(g) - window_size, STRIDE):
                X_win = gX[i:i + window_size]
                y_tgt = gy[i + window_size]  # 다음 시점 예측

                # 입력만 스케일
                X_scaled = scaler_X.transform(X_win).astype(DTYPE_X)

                X_memmap[out_idx] = X_scaled
                y_memmap[out_idx, 0] = y_tgt.astype(DTYPE_Y)
                out_idx += 1

        print(f"[{meter}] wrote {out_idx} windows → {X_path.name}, {y_path.name}")

    print("All meters processed.")


if __name__ == "__main__":
    main()
