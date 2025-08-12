import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

# 재현성(최소)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 설정
# input_size = 13          # X의 feature 수  # CHANGED: 아래에서 자동 감지
output_size = 1           # y 중 'value_log' 하나만 예측
hidden_size = 64
num_layers = 2
dropout = 0.2
batch_size = 64
num_epochs = 50
learning_rate = 0.001
target_index = 0          # y에서 "value_log" 위치
patience = 5              # Early Stopping 대기 epoch 수
min_delta = 1e-4          # 최소 개선 폭
MIN_EPOCHS = 15

device = torch.device("cpu")

# 경로 설정
X_DIR = Path("data/processed")
Y_DIR = Path("data/processed")
MODEL_PATH = Path("models/lstm_model_all.pth")
LOSS_PLOT_PATH = Path("results/loss_curve_all.png")
LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 실제 존재하는 meter_type만 필터링
def available_meter_types(x_dir, y_dir, all_types):
    available = []
    for m in all_types:
        x_path = x_dir / f"X_lstm_{m}.npy"
        y_path = y_dir / f"y_lstm_{m}.npy"
        if x_path.exists() and y_path.exists():
            available.append(m)
        else:
            print(f"Skipping meter_type={m} (missing file)")
    return available

all_meter_types = ["electricity", "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar"]
meter_types = available_meter_types(X_DIR, Y_DIR, all_meter_types)
print(f"Using meter types: {meter_types}")

# CHANGED: input_size 자동 감지 (첫 번째 존재하는 X 파일에서)
if len(meter_types) == 0:
    raise RuntimeError("No meter types found with X/y files.")
sample_X = np.load(X_DIR / f"X_lstm_{meter_types[0]}.npy", mmap_mode="r")
input_size = sample_X.shape[-1]
print(f"Detected input_size = {input_size}")

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 마지막 time step의 출력만 사용

# Custom Dataset 정의 (전체 meter_type 통합)
class NPYDataset(Dataset):
    def __init__(self, meter_types, x_dir, y_dir, target_index=0):
        self.X_data = [np.load(x_dir / f"X_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.y_data = [np.load(y_dir / f"y_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.target_index = target_index
        self.indices = [(i, j) for i, x in enumerate(self.X_data) for j in range(len(x))]

        # 각 미터 타입별 y 통계 (log 공간)
        self.stats = []
        for y in self.y_data:
            v = y[:, target_index]
            mu = float(np.mean(v))
            sd = float(np.std(v) + 1e-12)  # 0-div 방지
            self.stats.append({"mean": mu, "std": sd})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        x = torch.tensor(self.X_data[i][j], dtype=torch.float32)

        # y를 per-meter Z-스코어로 표준화해서 학습 (log 공간)
        y_raw = self.y_data[i][j][self.target_index]
        mu, sd = self.stats[i]["mean"], self.stats[i]["std"]
        y = torch.tensor((y_raw - mu) / sd, dtype=torch.float32)

        return x, y.unsqueeze(0)

meter_weights = {
    "electricity": 3.0,     # 가장 영향력 크고 데이터 많음
    "chilledwater": 2.5,    # 냉방 직접 영향, 데이터 충분
    "steam": 2.0,           # 난방 영향 크나 데이터 적음
    "hotwater": 1.5,        # 난방/온수 영향, 값 변동 큼
    "gas": 1.2,             # 변동폭 큼, 예측 불확실성 중간
    "water": 1.0,           # 부가 영향
    "irrigation": 0.8,      # 상대적으로 낮은 우선순위
    "solar": 0.5            # 보조 에너지원, 공급량 변동 큼
}

dataset = NPYDataset(meter_types, X_DIR, Y_DIR, target_index=target_index)

# 각 인덱스에 해당하는 meter_type 이름을 가져와서 가중치 리스트 생성
weights = []
for i, j in dataset.indices:
    mname = meter_types[i]
    weights.append(meter_weights.get(mname, 1.0))
weights = torch.DoubleTensor(weights)

# WeightedRandomSampler 생성
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=os.cpu_count() // 2,
    pin_memory=(device.type == "cuda")  # CPU면 False → 경고 제거
)

# 학습 준비
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)

# CHANGED: 손실함수는 HuberLoss 고정 (아래에서 MSE로 덮어쓰지 않음)
criterion = nn.HuberLoss(delta=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Early Stopping 변수
best_loss = float("inf")
best_epoch = 0
wait = 0
losses = []

# 학습 루프
print("Start training...")
stopped_early = False

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, dynamic_ncols=True)

    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)  # HuberLoss 사용
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix_str(f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
    losses.append(avg_loss)

    # Early Stopping 체크
    if avg_loss < best_loss - min_delta:
        best_loss = avg_loss
        best_epoch = epoch + 1
        wait = 0
        torch.save(model.state_dict(), MODEL_PATH)
        tqdm.write(f"Model improved. Saved to {MODEL_PATH} (epoch {best_epoch}, loss={best_loss:.4f})")
    else:
        wait += 1

    # 최소 에폭 보장 + patience 적용
    if (epoch + 1) >= MIN_EPOCHS and wait >= patience:
        tqdm.write(f"Early stopping at epoch {epoch+1} (best @ {best_epoch}, best loss={best_loss:.6f})")
        stopped_early = True
        break

if stopped_early:
    print(f"Training stopped early at epoch {epoch+1} (patience={patience}, best epoch={best_epoch})")
else:
    print(f"Training completed full {num_epochs} epochs (max epoch reached). Best epoch={best_epoch}")

# 손실 곡선 저장
plt.figure()
plt.plot(losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")  # CHANGED: 라벨만 정리
plt.grid()
plt.savefig(LOSS_PLOT_PATH)
plt.close()

print("Training complete.")
print(f"Best model saved to: {MODEL_PATH}")
print(f"Loss plot saved to: {LOSS_PLOT_PATH}")
