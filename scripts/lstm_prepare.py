import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
import json

# ===== 설정 =====
INPUT_PATH = Path("data/processed/office_data_merged.parquet")
OUTPUT_DIR = Path("data/processed")
SCALER_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)

# 시퀀스 파라미터
window_size = 36
STRIDE = 4

# dtype
DTYPE_X = "float16"
DTYPE_Y = "float32"

# 타깃 스케일
USE_LOG1P_TARGET = True

# 입력 컬럼 후보
TIME_COLS   = ["is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
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
    return (n_seq_est + stride - 1) // stride

def main():
    print(f"Loading: {INPUT_PATH}")
    df_all = pd.read_parquet(INPUT_PATH)

    # 필수 컬럼
    must_cols = ["timestamp", "building_id", "meter_type", "value"]
    for c in must_cols:
        if c not in df_all.columns:
            raise ValueError(f"Missing required column: {c}")

    # 정리
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["timestamp", "building_id", "meter_type", "value"]).sort_values("timestamp")

    # 누락 키 보완
    for col in ["source", "site_id"]:
        if col not in df_all.columns:
            df_all[col] = "NA"

    # building_id 카테고리화
    if df_all["building_id"].dtype.name != "category":
        df_all["building_id"] = df_all["building_id"].astype("category")

    # meter_type one-hot
    if df_all["meter_type"].dtype.name != "category":
        df_all["meter_type"] = df_all["meter_type"].astype("category")
    mt_dummies = pd.get_dummies(df_all["meter_type"], prefix="mt", dtype=np.int8)
    df_all = pd.concat([df_all, mt_dummies], axis=1)

    # 입력 컬럼 구성
    time_cols   = _safe_cols(df_all, TIME_COLS)
    weather_cols= _safe_cols(df_all, WEATHER_CANDIDATES)
    static_cols = _safe_cols(df_all, STATIC_CANDIDATES)
    mt_cols     = sorted([c for c in df_all.columns if c.startswith("mt_")])

    if USE_LOG1P_TARGET:
        df_all["value_log"] = np.log1p(np.clip(df_all["value"].astype("float32"), a_min=0, a_max=None))
        y_col = "value_log"
        value_feature_col = "value_log"
    else:
        y_col = "value"
        value_feature_col = "value"

    input_cols = time_cols + weather_cols + static_cols + mt_cols + [value_feature_col]
    print(f"Selected input_cols ({len(input_cols)}): {input_cols}")
    print(f"Target column: {y_col}")

    # 스케일러
    df_sample = df_all.dropna(subset=input_cols + [y_col]).sample(n=min(100_000, len(df_all)), random_state=42)
    scaler_X = StandardScaler().fit(df_sample[input_cols].values.astype("float32"))
    joblib.dump(scaler_X, SCALER_DIR / "scaler_X_all.pkl")
    print("Saved scaler_X_all.pkl")

    # 그룹핑 (건물 경계 유지)
    group_keys = ["source", "site_id", "building_id", "meter_type"]
    grouped = df_all.groupby(group_keys, observed=True, sort=False)

    meters = list(df_all["meter_type"].cat.categories)
    print(f"Meters found: {meters}")

    # 1 pass: 수량 계산
    total_per_meter = {m: 0 for m in meters}
    print("Counting windows per meter (pass 1)...")
    for (src, site, bid, mtype), g in tqdm(grouped, total=len(grouped)):
        g = g.dropna(subset=input_cols + [y_col]).sort_values("timestamp")
        n_out = _count_windows(len(g), window_size, STRIDE)
        if n_out > 0:
            total_per_meter[mtype] += n_out

    # building 라벨 저장
    (SCALER_DIR / "building_id_categories.json").write_text(
        json.dumps(list(df_all["building_id"].cat.categories.astype(str)), ensure_ascii=False, indent=2)
    )
    print("Saved building_id_categories.json")

    # 2 pass: 쓰기
    for meter in meters:
        total_out = total_per_meter.get(meter, 0)
        if total_out <= 0:
            print(f"[{meter}] no windows, skip.")
            continue

        X_path = OUTPUT_DIR / f"X_lstm_{meter}.npy"
        y_path = OUTPUT_DIR / f"y_lstm_{meter}.npy"
        B_META_PATH = OUTPUT_DIR / f"meta_building_{meter}.npy"
        T_META_PATH = OUTPUT_DIR / f"meta_ts_{meter}.npy"            # <<< 추가: 타깃 시점 timestamp

        X_memmap = np.lib.format.open_memmap(
            X_path, dtype=DTYPE_X, mode="w+",
            shape=(total_out, window_size, len(input_cols))
        )
        y_memmap = np.lib.format.open_memmap(
            y_path, dtype=DTYPE_Y, mode="w+",
            shape=(total_out, 1)
        )
        b_meta = np.lib.format.open_memmap(B_META_PATH, dtype="int32", mode="w+", shape=(total_out,))
        t_meta = np.lib.format.open_memmap(T_META_PATH, dtype="int64", mode="w+", shape=(total_out,))  # ns since epoch

        out_idx = 0
        print(f"[{meter}] writing {total_out} windows...")
        for (src, site, bid, mtype), g in grouped:
            if mtype != meter:
                continue

            g = g.dropna(subset=input_cols + [y_col]).sort_values("timestamp")
            n_out = _count_windows(len(g), window_size, STRIDE)
            if n_out <= 0:
                continue

            gX = g[input_cols].values.astype("float32")
            gy = g[y_col].values.astype("float32")
            b_codes = g["building_id"].cat.codes.values.astype("int32")
            # timestamp를 ns 정수로
            ts_ns = g["timestamp"].values.astype("datetime64[ns]").astype("int64")

            for i in range(0, len(g) - window_size, STRIDE):
                X_win = gX[i:i + window_size]
                y_tgt = gy[i + window_size]

                X_scaled = scaler_X.transform(X_win).astype(DTYPE_X)
                X_memmap[out_idx] = X_scaled
                y_memmap[out_idx, 0] = y_tgt.astype(DTYPE_Y)

                b_meta[out_idx] = int(b_codes[i + window_size])   # 타깃 시점 building
                t_meta[out_idx] = int(ts_ns[i + window_size])     # <<< 타깃 시점 timestamp(ns)

                out_idx += 1

        print(f"[{meter}] wrote {out_idx} windows → {X_path.name}, {y_path.name}, {B_META_PATH.name}, {T_META_PATH.name}")

    print("All meters processed.")

if __name__ == "__main__":
    main()
