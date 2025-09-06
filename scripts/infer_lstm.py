"""
infer_lstm.py  (minimal patch to write building_id into policy CSV)

- 기존 추론/메트릭 로직은 그대로
- 추가(패치):
  * meta_building_{meter}.npy 로드
  * models/building_id_categories.json 로드
  * policy CSV에 zone_id 대신 building_id를 기록
"""

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
from datetime import datetime, timedelta

# Config 학습과 동일
# input_size 모델과 동일
output_size = 1
hidden_size = 64
num_layers = 2
dropout = 0.2
batch_size = 256
target_index = 0

device = torch.device("cpu")  # CPU-only 환경 가정

# Paths
X_DIR = Path("data/processed")
Y_DIR = Path("data/processed")
MODEL_PATH = Path("models/lstm_model_all.pth")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = RESULTS_DIR / "preds_all.csv"
POLICY_CSV = RESULTS_DIR / "policy_events_ready.csv"  # 여기의 2번째 컬럼에 building_id를 씁니다
METRICS_CSV = RESULTS_DIR / "metrics_per_meter.csv"

# 메타/카테고리 경로
BUILDING_CATS_PATH = Path("models/building_id_categories.json")

# Utilities
def available_meter_types(x_dir: Path, y_dir: Path, all_types):
    available = []
    for m in all_types:
        if (x_dir / f"X_lstm_{m}.npy").exists() and (y_dir / f"y_lstm_{m}.npy").exists():
            available.append(m)
        else:
            print(f"Skipping meter_type={m} (missing file)")
    return available

ALL_TYPES = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]
METER_TYPES = available_meter_types(X_DIR, Y_DIR, ALL_TYPES)
print(f"Using meter types: {METER_TYPES}")

# CHANGED: input_size 자동 감지 (첫 가용 X에서)
if len(METER_TYPES) == 0:
    raise RuntimeError("No meter types found.")

# PATCH (권장): input_size 자동 감지 — 기존 하드코딩 대신 첫 X의 feature 수로 설정
_sample = np.load(X_DIR / f"X_lstm_{METER_TYPES[0]}.npy", mmap_mode="r")
input_size = int(_sample.shape[-1])
print(f"[info] inferred input_size = {input_size}")

# building 메타/라벨 로더
def _load_building_meta_and_labels(meter_types, x_dir: Path):
    b_meta_list = []
    for m in meter_types:
        p = x_dir / f"meta_building_{m}.npy"
        if not p.exists():
            print(f"[warn] building meta not found for meter={m}")
            b_meta_list.append(None)
        else:
            b_meta_list.append(np.load(p, mmap_mode="r"))
    if BUILDING_CATS_PATH.exists():
        building_labels = json.loads(BUILDING_CATS_PATH.read_text())
    else:
        building_labels = []
        print(f"[warn] building_id_categories.json not found.")
    return b_meta_list, building_labels

B_META_LIST, BUILDING_LABELS = _load_building_meta_and_labels(METER_TYPES, X_DIR)

# Model
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
    def __init__(self, meter_types, x_dir: Path, y_dir: Path, target_index=0, b_meta_list=None):
        self.meter_types = meter_types
        self.X_data = [np.load(x_dir / f"X_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.y_data = [np.load(y_dir / f"y_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.target_index = target_index
        self.indices = [(mi, sj) for mi, x in enumerate(self.X_data) for sj in range(len(x))]
        # building meta 주입
        self.b_meta = b_meta_list

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        mi, sj = self.indices[idx]
        x = torch.tensor(self.X_data[mi][sj], dtype=torch.float32)
        y = torch.tensor(self.y_data[mi][sj][self.target_index], dtype=torch.float32)  # 저장된 공간(보통 log)
        # building 코드 반환 (없으면 -1)
        b_code = -1
        if self.b_meta is not None and self.b_meta[mi] is not None:
            b_code = int(self.b_meta[mi][sj])
        return x, y.unsqueeze(0), mi, sj, b_code

# Load data/model
dataset = NPYDatasetInfer(METER_TYPES, X_DIR, Y_DIR, target_index=target_index, b_meta_list=B_META_LIST)
num_workers = max(os.cpu_count() // 2, 0)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
    pin_memory=False,
)

# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

# Stats for inverse-standardization (여기서는 y가 z-score(log) 저장됐다고 가정)
METER_STATS = []
METER_COUNTS = []
for m in METER_TYPES:
    y_arr = np.load(Y_DIR / f"y_lstm_{m}.npy", mmap_mode="r")[:, target_index]
    mu = float(np.mean(y_arr))
    sd = float(np.std(y_arr) + 1e-12)
    METER_STATS.append({"mean": mu, "std": sd})
    METER_COUNTS.append(len(y_arr))

N_total = sum(METER_COUNTS)
overall_mean = (
    sum(METER_STATS[i]["mean"] * METER_COUNTS[i] for i in range(len(METER_TYPES))) / max(N_total, 1)
)

# Inference & Metrics
overall_mse_sum = 0.0
overall_mae_sum = 0.0
overall_sst = 0.0
overall_n = 0

per_meter = {
    i: {"name": METER_TYPES[i], "mse_sum": 0.0, "mae_sum": 0.0, "sse": 0.0, "sst": 0.0, "n": 0}
    for i in range(len(METER_TYPES))
}

pred_file = open(PRED_CSV, "w", newline="")
pred_writer = csv.writer(pred_file)
pred_writer.writerow(["meter_type", "seq_index", "y_true", "y_pred"])

policy_file = open(POLICY_CSV, "w", newline="")
policy_writer = csv.writer(policy_file)
# 두 번째 컬럼명을 building_id로 변경
policy_writer.writerow(["timestamp", "building_id", "meter_type", "value", "indoor_temperature_pred", "occupancy_pred", "horizon"])

Y_STANDARDIZED = True  # 학습 시 y를 z-score(log)로 썼다면 True

# 타임스탬프 합성(필요 시 수정)
START_TIME = datetime.utcnow()
STEP_MINUTES = 1

with torch.inference_mode():
    for X, y, mi, sj, b_code in tqdm(loader, desc="Batches"):
        X = X.to(device)
        pred = model(X).cpu().squeeze(1)   # (B,) z-score in log space
        y = y.cpu().squeeze(1)             # (B,) log space

        # inverse standardization (log space)
        if Y_STANDARDIZED:
            pred_log = pred.clone()
            for k in range(len(mi)):
                idx = int(mi[k].item())
                mu = METER_STATS[idx]["mean"]
                sd = METER_STATS[idx]["std"]
                pred_log[k] = pred[k] * sd + mu
        else:
            pred_log = pred

        # log -> linear 복원
        y_lin = np.expm1(y.numpy())
        pred_lin = np.expm1(pred_log.numpy())

        # metrics 누적 (linear space)
        err = pred_lin - y_lin
        overall_mse_sum += float((err ** 2).sum())
        overall_mae_sum += float(np.abs(err).sum())
        overall_sst += float(((y_lin - np.expm1(overall_mean)) ** 2).sum())
        overall_n += y_lin.size

        for k in range(len(mi)):
            idx = int(mi[k].item())
            mu_lin = float(np.expm1(METER_STATS[idx]["mean"]))
            e2 = float((pred_lin[k] - y_lin[k]) ** 2)
            ae = float(abs(pred_lin[k] - y_lin[k]))
            per_meter[idx]["mse_sum"] += e2
            per_meter[idx]["mae_sum"] += ae
            per_meter[idx]["sse"] += e2
            per_meter[idx]["sst"] += float((y_lin[k] - mu_lin) ** 2)
            per_meter[idx]["n"] += 1

        # write CSVs (linear)
        for k in range(len(mi)):
            meter_type = METER_TYPES[int(mi[k].item())]
            seq_index = int(sj[k].item())
            y_true_val = float(y_lin[k])
            y_pred_val = float(pred_lin[k])

            pred_writer.writerow([meter_type, seq_index, y_true_val, y_pred_val])

            # building 코드 → 라벨 복원
            bc = int(b_code[k].item())
            if 0 <= bc < len(BUILDING_LABELS):
                building_label = str(BUILDING_LABELS[bc])
            else:
                building_label = "unknown_building"

            ts_val = (START_TIME + timedelta(minutes=seq_index * STEP_MINUTES)).isoformat() + "Z"
            horizon = seq_index * STEP_MINUTES

            # policy CSV에 building_id 기록
            policy_writer.writerow([ts_val, building_label, meter_type, y_pred_val, "", "", horizon])

# close files
pred_file.close()
policy_file.close()

# Final metrics (linear)
overall_mse = overall_mse_sum / max(overall_n, 1)
overall_rmse = math.sqrt(overall_mse)
overall_mae = overall_mae_sum / max(overall_n, 1)
overall_r2 = 1.0 - (overall_mse_sum / max(overall_sst, 1e-12))

print(f"OVERALL  MSE:{overall_mse:.6f} | RMSE:{overall_rmse:.6f} | MAE:{overall_mae:.6f} | R2:{overall_r2:.4f}")
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
            std_lin = float(np.std(np.expm1(np.load(Y_DIR / f"y_lstm_{METER_TYPES[i]}.npy", mmap_mode="r")[:, target_index])))
            sse = per_meter[i]["sse"]
            sst = per_meter[i]["sst"]
            nrmse = rmse / (std_lin if std_lin > 0 else float("nan"))
            r2 = 1.0 - (sse / max(sst, 1e-12))
        writer.writerow([per_meter[i]["name"], n, f"{mse:.6f}", f"{rmse:.6f}", f"{mae:.6f}", f"{nrmse:.3f}", f"{r2:.4f}"])
        print(f"{per_meter[i]['name']:>12s} | n={n:7d} | RMSE:{rmse:.6f} | MAE:{mae:.6f} | NRMSE:{nrmse:.3f} | R2:{r2:.3f}")

print(f"Predictions saved to: {PRED_CSV}")
print(f"Policy events saved to: {POLICY_CSV}")
print(f"Per-meter metrics  : {METRICS_CSV}")

(Path(RESULTS_DIR) / "meter_stats.json").write_text(
    json.dumps([
        {"meter": m, **s} for m, s in zip(METER_TYPES, METER_STATS)
    ], ensure_ascii=False, indent=2)
)
print(f"Meter stats saved to: {RESULTS_DIR / 'meter_stats.json'}")
