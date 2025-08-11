import pandas as pd
from pathlib import Path
import traceback
from fastparquet import ParquetFile

# 표준 컬럼 정의
STANDARD_COLUMNS = [
    "timestamp", "building_id", "meter_type", "value", "source",
    "site_id", "primary_use", "square_feet", "year_built"
]

# 경로 설정
ASHRAE = Path("data/raw/ashrae-energy-prediction")
BDG2 = Path("data/raw/buildingdatagenomeproject2")
BDG2_META = BDG2 / "metadata.csv"
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)
MERGED_PATH = PROCESSED / "merged_energy.parquet"

def clean_and_standardize_metadata(df, source):
    rename_dict = {
        "yearbuilt": "year_built",
        "sqft": "square_feet",
        "numberoffloors": "floor_count",
        "primaryspaceusage": "primary_use"
    }
    df = df.rename(columns=rename_dict)
    df["building_id"] = df["building_id"].astype(str)
    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str)
    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].astype(str)
    return df[[c for c in df.columns if c in STANDARD_COLUMNS or c == "building_id"]]

def validate_parquet(path):
    try:
        ParquetFile(path)
        return True
    except Exception as e:
        print(f"[Invalid Parquet] {path.name}: {e}")
        return False

def load_ashrae():
    print("Loading ASHRAE dataset")
    df = pd.read_csv(ASHRAE / "train.csv", usecols=["timestamp", "building_id", "meter", "meter_reading"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"meter": "meter_type", "meter_reading": "value"})
    df["building_id"] = df["building_id"].astype(str)
    df["meter_type"] = df["meter_type"].astype(str)
    METER_TYPE_MAP = {
        '0': 'electricity',
        '1': 'chilledwater',
        '2': 'steam',
        '3': 'hotwater'
    }
    df["meter_type"] = df["meter_type"].map(METER_TYPE_MAP)
    df["source"] = "ASHRAE"

    meta_path = ASHRAE / "building_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta = clean_and_standardize_metadata(meta, "ASHRAE")
        df = pd.merge(df, meta, on="building_id", how="left")
        print("ASHRAE metadata merged.")
        
    # 필터링: primary_use가 있을 경우 "office"만 남김
    if "primary_use" in df.columns and df["primary_use"].notna().any():
        df = df[df["primary_use"].str.lower() == "office"]

    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str)
    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].astype(str)

    output_path = PROCESSED / "merging_ashrae.parquet"
    df.to_parquet(output_path, index=False, engine="fastparquet", compression="snappy")
    print(f"[Saved] ASHRAE → {output_path.name} | rows: {len(df)}")
    return output_path

# def process_weather_csv():
#     file_path = BDG2 / "weather.csv"
#     if not file_path.exists():
#         print("[Missing] weather.csv")
#         return None

#     try:
#         df = pd.read_csv(file_path)
#         df["timestamp"] = pd.to_datetime(df["timestamp"])
#         df["site_id"] = df["site_id"].astype(str)

#         df = pd.melt(
#             df,
#             id_vars=["timestamp", "site_id"],
#             var_name="meter_type",
#             value_name="value"
#         )
#         df["building_id"] = "external"
#         df["source"] = "BDG2"

#         # 누락된 column 채우기
#         for col in ["primary_use", "square_feet", "year_built"]:
#             df[col] = None

#         df = df[[c for c in STANDARD_COLUMNS if c in df.columns]]
#         output_path = PROCESSED / "merging_bdg2_weather.parquet"
#         df.to_parquet(output_path, index=False, engine="fastparquet", compression="snappy")

#         if validate_parquet(output_path):
#             print(f"[Saved] {output_path.name} | rows: {len(df)}")
#             return output_path
#         else:
#             print(f"[Invalid] {output_path.name}")
#             return None

#     except Exception as e:
#         print(f"[Error] Failed to process weather.csv: {e}")
#         traceback.print_exc()
#         return None


def process_wide_meter_file_in_chunks(meter_type, filename, chunk_size=530):
    file_path = BDG2 / filename
    if not file_path.exists():
        print(f"[Missing] {filename}")
        return []

    print(f"Reading {filename} in {chunk_size}-column chunks")
    meta = pd.read_csv(BDG2_META) if BDG2_META.exists() else None
    if meta is not None:
        meta = clean_and_standardize_metadata(meta, "BDG2")

    all_columns = pd.read_csv(file_path, nrows=1).columns.tolist()
    building_cols = [c for c in all_columns if c != "timestamp"]
    total_chunks = (len(building_cols) + chunk_size - 1) // chunk_size

    saved_paths = []

    for i in range(total_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(building_cols))
        cols = ["timestamp"] + building_cols[start:end]
        try:
            df = pd.read_csv(file_path, usecols=cols, low_memory=False)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.melt(id_vars=["timestamp"], var_name="building_id", value_name="value")
            df["building_id"] = df["building_id"].astype(str)
            df["meter_type"] = meter_type
            df["source"] = "BDG2"
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

            if meta is not None:
                df = pd.merge(df, meta, on="building_id", how="left")

            for col in ["site_id", "year_built"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            chunk_path = PROCESSED / f"merging_bdg2_{meter_type}_chunk_{i+1}.parquet"
            df = df[[c for c in df.columns if c in STANDARD_COLUMNS]]
            df.to_parquet(chunk_path, index=False, engine="fastparquet", compression="snappy")

            if validate_parquet(chunk_path):
                saved_paths.append(chunk_path)
                print(f"[Saved] {chunk_path.name} | rows: {len(df)}")
        except Exception as e:
            print(f"[Error] Failed chunk {i+1}/{total_chunks}: {e}")
            traceback.print_exc()

    return saved_paths

def process_and_save_bdg2():
    print("Processing BDG2 meter files")
    meter_files = {
        "electricity": "electricity.csv",
        "gas": "gas_cleaned.csv",
        "chilledwater": "chilledwater_cleaned.csv",
        "hotwater": "hotwater_cleaned.csv",
        "steam": "steam_cleaned.csv",
        "water": "water_cleaned.csv",
        "irrigation": "irrigation_cleaned.csv",
        "solar": "solar_cleaned.csv"
    }

    saved_paths = []
    for meter_type, filename in meter_files.items():
        if meter_type == "electricity":
            saved_paths += process_wide_meter_file_in_chunks(meter_type, filename)
            continue

        file_path = BDG2 / filename
        if not file_path.exists():
            print(f"[Missing] {filename}")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.melt(id_vars=["timestamp"], var_name="building_id", value_name="value")
            df["building_id"] = df["building_id"].astype(str)
            df["meter_type"] = meter_type
            df["source"] = "BDG2"
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

            if BDG2_META.exists():
                meta = pd.read_csv(BDG2_META)
                meta = clean_and_standardize_metadata(meta, "BDG2")
                df = pd.merge(df, meta, on="building_id", how="left")

            for col in ["site_id", "year_built"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            output_path = PROCESSED / f"merging_bdg2_{meter_type}.parquet"
            df = df[[c for c in df.columns if c in STANDARD_COLUMNS]]
            df.to_parquet(output_path, index=False, engine="fastparquet", compression="snappy")
            if validate_parquet(output_path):
                saved_paths.append(output_path)
                print(f"[Saved] {output_path.name} | rows: {len(df)}")

        except Exception as e:
            print(f"[Error] Failed to process {meter_type}: {e}")
            traceback.print_exc()

    # # 추가: 날씨 처리
    # weather_path = process_weather_csv()
    # if weather_path:
    #     saved_paths.append(weather_path)

    return saved_paths

def merge_parquet_files_streaming(parquet_paths):
    print("Merging parquet files")
    if MERGED_PATH.exists():
        MERGED_PATH.unlink()

    total_rows = 0
    for path in parquet_paths:
        try:
            if not validate_parquet(path):
                continue
            df = pd.read_parquet(path)
            if "value" in df.columns:
                df.dropna(subset=["value"], inplace=True)
            df = df[[col for col in STANDARD_COLUMNS if col in df.columns]]

            # primary_use가 있는 경우에만 "office" 필터 적용
            if "primary_use" in df.columns and df["primary_use"].notna().any():
                df = df[df["primary_use"].str.lower() == "office"]
            if df.empty:
                continue
            
            df.to_parquet(
                MERGED_PATH,
                index=False,
                engine="fastparquet",
                compression="snappy",
                append=MERGED_PATH.exists()
            )
            print(f"[Appended] {path.name} | rows: {len(df)}")
            total_rows += len(df)
        except Exception as e:
            print(f"[Error] Skipped {path.name}: {e}")
            traceback.print_exc()

    print(f"[Merged] Total rows: {total_rows:,}")
    print(f"[Output] {MERGED_PATH.name}")

def merge_datasets():
    print("Running full pipeline")
    ashrae_path = load_ashrae()
    bdg2_paths = process_and_save_bdg2()
    all_paths = [ashrae_path] + bdg2_paths
    merge_parquet_files_streaming(all_paths)

if __name__ == "__main__":
    merge_datasets()
