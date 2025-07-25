import pandas as pd
from pathlib import Path

ASHRAE = Path("data/raw/ashrae-energy-prediction")
BDG2 = Path("data/raw/buildingdatagenomeproject2")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_ashrae():
    print("Loading ASHRAE dataset...")
    df = pd.read_csv(ASHRAE / "train.csv", usecols=["timestamp", "building_id", "meter", "meter_reading"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={
        "meter": "meter_type",
        "meter_reading": "value"
    })
    df["building_id"] = df["building_id"].astype(str)
    df["meter_type"] = df["meter_type"].astype(str)
    df["source"] = "ASHRAE"
    return df

def load_bdg2():
    print("Loading BDG2 dataset...")
    meter_files = {
        "electricity": "electricity.csv",
        "gas": "gas.csv",
        "chilledwater": "chilledwater.csv",
        "hotwater": "hotwater.csv",
        "steam": "steam.csv"
    }
    all_data = []

    for meter_type, filename in meter_files.items():
        print(f"Reading {filename}...")
        df = pd.read_csv(BDG2 / filename)
        df = df.rename(columns={"local_15min": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.melt(id_vars=["timestamp"], var_name="building_id", value_name="value")
        df["building_id"] = df["building_id"].astype(str)
        df["meter_type"] = meter_type
        df["source"] = "BDG2"
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def merge_datasets():
    df1 = load_ashrae()
    df2 = load_bdg2()

    print("Concatenating datasets...")
    df = pd.concat([df1, df2], ignore_index=True)

    print("Dropping nulls...")
    df.dropna(subset=["value"], inplace=True)

    print("Saving merged dataset without sort (to avoid memory overload)...")
    df.to_parquet(PROCESSED / "merged_energy.parquet", index=False)

    print("✅ Done. File saved to data/processed/merged_energy.parquet")

if __name__ == "__main__":
    merge_datasets()
