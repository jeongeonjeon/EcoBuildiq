# Data Flow
#
# train.csv            -->   meter_reading 등
# building_metadata.csv -->  primary_use 포함
# weather_train.csv    -->   날씨 정보
#          ↓ merge
#        df (전체)
#          ↓ filter
#   df[df['primary_use'] == 'Office']

import pandas as pd
from pathlib import Path

ASHRAW = Path("data/raw/ashrae-energy-prediction")
PROCESSED = Path("data/processed")

def preprocess():
    print("Loading raw CSV files...")
    train = pd.read_csv(ASHRAW / "train.csv")
    meta = pd.read_csv(ASHRAW / "building_metadata.csv")
    weather = pd.read_csv(ASHRAW / "weather_train.csv")

    print("Parsing datetime...")
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])

    print("Merging datasets...")
    df = train.merge(meta, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    print("Filtering for Office buildings...")
    df = df[df['primary_use'] == 'Office']

    print("Filling missing values...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("Extracting time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    print("Saving preprocessed data...")
    df.to_parquet(PROCESSED / "office_data.parquet", index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
