import pandas as pd
from pathlib import Path
from fastparquet import ParquetFile

# 기본 설정
STANDARD_COLUMNS = [
    "timestamp", "building_id", "meter_type", "value", "source",
    "site_id", "primary_use", "square_feet", "year_built"
]

ASHRAE_DIR = Path("data/raw/ashrae-energy-prediction")
BDG2_DIR = Path("data/raw/buildingdatagenomeproject2")
BDG2_META_PATH = BDG2_DIR / "metadata.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MERGED_PARQUET = PROCESSED_DIR / "merged_energy.parquet"

def standardize_metadata(df):
    df = df.rename(columns={
        "yearbuilt": "year_built",
        "sqft": "square_feet",
        "numberoffloors": "floor_count",
        "primaryspaceusage": "primary_use"
    })
    df["building_id"] = df["building_id"].astype(str)
    if "year_built" in df.columns:
        df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce").astype("Int64")
    return df[[col for col in df.columns if col in STANDARD_COLUMNS or col == "building_id"]]

def is_valid_parquet(path):
    try:
        ParquetFile(path)
        return True
    except:
        return False

def get_office_ids(metadata):
    if "primary_use" in metadata.columns:
        return metadata[metadata["primary_use"].str.lower() == "office"]["building_id"].astype(str).tolist()
    return []

def process_ashrae():
    df = pd.read_csv(ASHRAE_DIR / "train.csv", usecols=["timestamp", "building_id", "meter", "meter_reading"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["building_id"] = df["building_id"].astype(str)
    df["meter_type"] = df["meter"].astype(str).map({
        '0': 'electricity', '1': 'chilledwater', '2': 'steam', '3': 'hotwater'
    })
    df["value"] = df["meter_reading"]
    df["source"] = "ASHRAE"
    df.drop(columns=["meter", "meter_reading"], inplace=True)

    meta_path = ASHRAE_DIR / "building_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta = standardize_metadata(meta)
        meta["building_id"] = meta["building_id"].astype(str)
        df = df.merge(meta, on="building_id", how="left")
        df = df[df["primary_use"].str.lower() == "office"]

    out_path = PROCESSED_DIR / "ashrae.parquet"
    df.to_parquet(out_path, index=False, compression="snappy", engine="fastparquet")
    return out_path

def process_bdg2():
    if not BDG2_META_PATH.exists():
        return []

    meta = pd.read_csv(BDG2_META_PATH)
    meta = standardize_metadata(meta)
    meta["building_id"] = meta["building_id"].astype(str)
    office_ids = get_office_ids(meta)
    processed_paths = []

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

    for meter, file in meter_files.items():
        file_path = BDG2_DIR / file
        if not file_path.exists():
            continue

        usecols = ["timestamp"] + office_ids
        try:
            df = pd.read_csv(file_path, usecols=lambda c: c in usecols, low_memory=False)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.melt(id_vars=["timestamp"], var_name="building_id", value_name="value")
            df["meter_type"] = meter
            df["source"] = "BDG2"
            df = df.merge(meta, on="building_id", how="left")
            df = df[[col for col in STANDARD_COLUMNS if col in df.columns]]
            out_path = PROCESSED_DIR / f"bdg2_{meter}.parquet"
            df.to_parquet(out_path, index=False, compression="snappy", engine="fastparquet")
            processed_paths.append(out_path)
        except Exception as e:
            print(f"Skipped {file}: {e}")

    return processed_paths

def merge_parquets(parquet_files):
    if MERGED_PARQUET.exists():
        MERGED_PARQUET.unlink()

    total = 0
    for path in parquet_files:
        if not is_valid_parquet(path):
            continue
        df = pd.read_parquet(path)
        df = df.dropna(subset=["value"])
        if "primary_use" in df.columns and df["primary_use"].notna().any():
            df = df[df["primary_use"].str.lower() == "office"]
        if df.empty:
            continue
        df = df[[col for col in STANDARD_COLUMNS if col in df.columns]]
        if "site_id" in df.columns:
            df["site_id"] = df["site_id"].astype(str)

        df.to_parquet(MERGED_PARQUET, index=False, compression="snappy", engine="fastparquet", append=MERGED_PARQUET.exists())
        total += len(df)

    print(f"Merged total rows: {total}")

def merge_datasets():
    ashrae = process_ashrae()
    bdg2 = process_bdg2()
    merge_parquets([ashrae] + bdg2)

if __name__ == "__main__":
    merge_datasets()
