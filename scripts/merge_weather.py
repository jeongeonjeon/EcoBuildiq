import pandas as pd
from pathlib import Path

# 경로 설정
DATA_PATH = Path("data/processed/merged_energy.parquet")
ASHRAE_WEATHER_PATH = Path("data/raw/ashrae-energy-prediction/weather_train.csv")
BDG2_WEATHER_PATH = Path("data/raw/buildingdatagenomeproject2/weather.csv")
OUTPUT_PATH = Path("data/processed/merged_energy_weather.parquet")

def load_weather_ashrae():
    df = pd.read_csv(ASHRAE_WEATHER_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["site_id"] = df["site_id"].astype(str)
    return df

def load_weather_bdg2():
    df = pd.read_csv(BDG2_WEATHER_PATH)
    df = df.rename(columns={
        "airTemperature": "air_temperature",
        "cloudCoverage": "cloud_coverage",
        "dewTemperature": "dew_temperature",
        "precipDepth1HR": "precip_depth_1_hr",
        "seaLvlPressure": "sea_level_pressure",
        "windDirection": "wind_direction",
        "windSpeed": "wind_speed"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["site_id"] = df["site_id"].astype(str)
    return df

def merge_weather():
    print("Loading merged energy data")
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["site_id"] = df["site_id"].astype(str)

    print("Splitting ASHRAE and BDG2")
    ashrae_df = df[df["source"] == "ASHRAE"].copy()
    bdg2_df = df[df["source"] == "BDG2"].copy()

    print("Merging ASHRAE weather")
    weather_ashrae = load_weather_ashrae()
    ashrae_df = ashrae_df.merge(weather_ashrae, on=["site_id", "timestamp"], how="left")

    print("Merging BDG2 weather")
    weather_bdg2 = load_weather_bdg2()
    bdg2_df = bdg2_df.merge(weather_bdg2, on=["site_id", "timestamp"], how="left")

    print("Combining ASHRAE and BDG2")
    combined_df = pd.concat([ashrae_df, bdg2_df], ignore_index=True)

    print(f"Saving merged dataset with weather → {OUTPUT_PATH}")
    combined_df.to_parquet(OUTPUT_PATH, index=False)
    print("Weather merge complete.")

if __name__ == "__main__":
    merge_weather()
