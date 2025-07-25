import numpy as np

def inverse_transform(array, scaler, column_index=-1):
    dummy = np.zeros((array.shape[0], scaler.mean_.shape[0]))
    dummy[:, column_index] = array.flatten()
    return scaler.inverse_transform(dummy)[:, column_index].reshape(-1, 1)
