import numpy as np

X = np.load("data/processed/X_lstm.npy")
y = np.load("data/processed/y_lstm.npy")

# NaN 제거
X = np.nan_to_num(X, nan=0.0)

print("X has NaN:", np.isnan(X).any())
print("y has NaN:", np.isnan(y).any())
print("X has Inf:", np.isinf(X).any())
print("y has Inf:", np.isinf(y).any())
print("y min/max:", y.min(), y.max())