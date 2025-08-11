import numpy as np
import pandas as pd

# meter_type 매핑 사전
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

# 역매핑: ID → 이름
reverse_meter_type_mapping = {v: k for k, v in meter_type_mapping.items()}

def meter_name_to_id(name):
    """문자열 meter_type 이름 → 숫자 ID"""
    return meter_type_mapping.get(name)

def meter_id_to_name(id_):
    """숫자 meter_type ID → 문자열 이름"""
    return reverse_meter_type_mapping.get(id_)

def decode_meter_type(scaled_value):
    """
    스케일된 meter_type 값을 원래 문자열로 복원
    주의: scaler.inverse_transform을 먼저 적용해야 함
    """
    # 가장 가까운 정수로 반올림하여 매핑
    id_ = int(round(scaled_value))
    return meter_id_to_name(id_)

def inverse_transform(array, scaler, column_index=-1):
    """
    단일 column에 대해 역정규화를 수행
    Parameters:
        array: (N, 1) or (N,) shape
        scaler: sklearn StandardScaler 객체
        column_index: scaler에서 해당 컬럼의 위치 (예: 'value'의 인덱스)
    Returns:
        원래 값으로 복원된 array
    """
    dummy = np.zeros((array.shape[0], scaler.mean_.shape[0]))
    dummy[:, column_index] = array.flatten()
    return scaler.inverse_transform(dummy)[:, column_index].reshape(-1, 1)
