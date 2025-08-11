import pandas as pd

# path = "data/raw/ashrae-energy-prediction/weather_train.csv"
# path = "data/raw/buildingdatagenomeproject2/weather.csv"
# df = pd.read_csv(path)

path = "data/processed/office_data_merged.parquet"
df = pd.read_parquet(path)


# 핵심 정보 출력
print("총 행 수:", len(df))
print("컬럼 목록:", df.columns.tolist())
print(df.dtypes)

# 누락 데이터 확인
print("\n결측치 개수:")
print(df.isnull().sum())

# 고유 meter_type 목록
print("\nmeter_type 종류:", df['meter_type'].unique())

# 표본 출력
print(f"{path} 샘플 데이터:")
print(df.sample(60, random_state=0))
