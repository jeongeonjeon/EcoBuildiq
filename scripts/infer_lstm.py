import csv
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import math
import json

# Config (학습과 동일하게)
input_size = 13
output_size = 1
hidden_size = 64
num_layers = 2
dropout = 0.2
batch_size = 256  # 128->256 변경(128~512)
target_index = 0

device = torch.device("cpu")

# Paths
X_DIR = Path("data/processed")
Y_DIR = Path("data/processed")
MODEL_PATH = Path("models/lstm_model_all.pth")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = RESULTS_DIR / "preds_all.csv"
METRICS_CSV = RESULTS_DIR / "metrics_per_meter.csv"


# Utilities
def available_meter_types(x_dir: Path, y_dir: Path, all_types):
    available = []
    for m in all_types:
        x_path = x_dir / f"X_lstm_{m}.npy"
        y_path = y_dir / f"y_lstm_{m}.npy"
        if x_path.exists() and y_path.exists():
            available.append(m)
        else:
            print(f"Skipping meter_type={m} (missing file)")
    return available

ALL_TYPES = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]
METER_TYPES = available_meter_types(X_DIR, Y_DIR, ALL_TYPES)
print(f"Using meter types: {METER_TYPES}")

# Model (학습과 동일)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Dataset (추론용으로 인덱스/미터타입 반환)
class NPYDatasetInfer(Dataset):
    def __init__(self, meter_types, x_dir: Path, y_dir: Path, target_index=0):
        self.meter_types = meter_types
        self.X_data = [np.load(x_dir / f"X_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.y_data = [np.load(y_dir / f"y_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.target_index = target_index
        # (meter_idx, sample_idx)
        self.indices = [(mi, sj) for mi, x in enumerate(self.X_data) for sj in range(len(x))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        mi, sj = self.indices[idx]
        x = torch.tensor(self.X_data[mi][sj], dtype=torch.float32)
        y = torch.tensor(self.y_data[mi][sj][self.target_index], dtype=torch.float32)
        return x, y.unsqueeze(0), mi, sj  # (B, T, F), (B,1), meter_idx, seq_idx

# Load data/model
dataset = NPYDatasetInfer(METER_TYPES, X_DIR, Y_DIR, target_index=target_index)
num_workers = max(os.cpu_count() // 2, 0)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,            # 추론은 순서 고정
    num_workers=num_workers,            # 재현성/안정성
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
    pin_memory=False
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

METER_STATS = []  # list of dicts with mean/std per meter
METER_COUNTS = []
for m in METER_TYPES:

    y_arr = np.load(Y_DIR / f"y_lstm_{m}.npy", mmap_mode="r")[:, target_index]
    mu = float(np.mean(y_arr))
    sd = float(np.std(y_arr) + 1e-12)  # avoid zero

    METER_STATS.append({"mean": mu, "std": sd})
    METER_COUNTS.append(len(y_arr))

N_total = sum(METER_COUNTS)
overall_mean = (
    sum(METER_STATS[i]["mean"] * METER_COUNTS[i] for i in range(len(METER_TYPES))) / max(N_total, 1)
)

# Inference + Metrics
overall_mse_sum = 0.0
overall_mae_sum = 0.0
overall_sst = 0.0
overall_n = 0

# per-meter 누적
per_meter = {
    i: {"name": METER_TYPES[i], "mse_sum": 0.0, "mae_sum": 0.0,"sse": 0.0,"sst": 0.0, "n": 0}
    for i in range(len(METER_TYPES))
}

# preds csv 작성
with open(PRED_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["meter_type", "seq_index", "y_true", "y_pred"])

    with torch.inference_mode():
        for X, y, mi, sj in tqdm(loader, desc="Batches"):
            X = X.to(device)
            pred = model(X).cpu().squeeze(1)   # (B,)
            y = y.cpu().squeeze(1)             # (B,)
            
            # 학습에서 y를 z-score로 썼다면, 여기서 역변환
            Y_STANDARDIZED = True  # <- 학습에서 표준화 적용했으니 True

            if Y_STANDARDIZED:
                pred_raw = pred.clone()
                for k in range(len(mi)):
                    idx = int(mi[k].item())
                    mu = METER_STATS[idx]["mean"]
                    sd = METER_STATS[idx]["std"]
                    pred_raw[k] = pred[k] * sd + mu
                pred = pred_raw  # 이후 err/CSV는 pred 기준

            # 누적 (overall)
            err = pred - y
            overall_mse_sum += (err ** 2).sum().item()
            overall_mae_sum += err.abs().sum().item()
            overall_sst += ((y - overall_mean) ** 2).sum().item()
            overall_n += y.numel()

            # 누적 (per-meter)
            for k in range(len(mi)):
                idx = int(mi[k].item())
                mu = METER_STATS[idx]["mean"]
                e2 = float((pred[k] - y[k]) ** 2)
                ae = float(abs(pred[k] - y[k]))
                per_meter[idx]["mse_sum"] += e2
                per_meter[idx]["mae_sum"] += ae
                per_meter[idx]["sse"] += e2
                per_meter[idx]["sst"] += float((y[k] - mu) ** 2)
                per_meter[idx]["n"] += 1

            # CSV 라인 기록
            for k in range(len(mi)):
                writer.writerow([
                    METER_TYPES[int(mi[k].item())],
                    int(sj[k].item()),
                    float(y[k].item()),
                    float(pred[k].item())
                ])

# 결과 출력/저장
overall_mse = overall_mse_sum / max(overall_n, 1)
overall_rmse = math.sqrt(overall_mse)
overall_mae = overall_mae_sum / max(overall_n, 1)
overall_r2 = 1.0 - (overall_mse_sum / max(overall_sst, 1e-12))

print(f"OVERALL  MSE:{overall_mse:.6f} | RMSE:{overall_rmse:.6f} | MAE:{overall_mae:.6f} | R2:{overall_r2:.4f}")
# per-meter CSV
with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["meter_type", "count", "MSE", "RMSE", "MAE", "NRMSE", "R2"])
    for i in range(len(METER_TYPES)):
        n = per_meter[i]["n"]
        if n == 0:
            mse = rmse = mae = nrmse = r2 = float("nan")
        else:
            mse = per_meter[i]["mse_sum"] / n
            rmse = math.sqrt(mse)
            mae = per_meter[i]["mae_sum"] / n
            std = METER_STATS[i]["std"]
            sse = per_meter[i]["sse"]
            sst = per_meter[i]["sst"]
            nrmse = rmse / std
            r2 = 1.0 - (sse / max(sst, 1e-12))
        writer.writerow([per_meter[i]["name"], n, f"{mse:.6f}", f"{rmse:.6f}", f"{mae:.6f}", f"{nrmse:.6f}", f"{r2:.4f}"])
        print(f"{per_meter[i]['name']:>12s} | n={n:7d} | RMSE:{rmse:.6f} | MAE:{mae:.6f} | NRMSE:{nrmse:.3f} | R2:{r2:.3f}")

print(f"Predictions saved to: {PRED_CSV}")
print(f"Per-meter metrics  : {METRICS_CSV}")


(Path(RESULTS_DIR)/"meter_stats.json").write_text(
    json.dumps([{"meter": m, **s} for m, s in zip(METER_TYPES, METER_STATS)], ensure_ascii=False, indent=2)
)
print(f"Meter stats saved to: {RESULTS_DIR/'meter_stats.json'}")
