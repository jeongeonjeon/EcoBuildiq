import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
input_size = 13          # X의 feature 수
output_size = 1          # y 중 'value' 하나만 예측
hidden_size = 64
num_layers = 2
dropout = 0.2
batch_size = 64
num_epochs = 30
learning_rate = 0.001
target_index = 0  # y에서 "value" 위치
patience = 3      # Early Stopping 대기 epoch 수
min_delta = 5e-4  # 최소 개선 폭, 1e-4 → 5e-4로 상향

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
# --- 기존 NPYDataset 정의에 추가/수정 ---

class NPYDataset(Dataset):
    def __init__(self, meter_types, x_dir, y_dir, target_index=0):
        self.X_data = [np.load(x_dir / f"X_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.y_data = [np.load(y_dir / f"y_lstm_{m}.npy", mmap_mode="r") for m in meter_types]
        self.target_index = target_index
        self.indices = [(i, j) for i, x in enumerate(self.X_data) for j in range(len(x))]

        # 각 미터 타입별 y 통계
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

        # y를 per-meter Z-스코어로 표준화해서 학습
        y_raw = self.y_data[i][j][self.target_index]
        mu, sd = self.stats[i]["mean"], self.stats[i]["std"]
        y = torch.tensor((y_raw - mu) / sd, dtype=torch.float32)

        return x, y.unsqueeze(0)



# DataLoader
dataset = NPYDataset(meter_types, X_DIR, Y_DIR, target_index=target_index)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count() // 2,  # CPU 절반 사용 (안정성 위해)
    pin_memory=True
)

# 학습 준비
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Early Stopping 변수
best_loss = float("inf")
best_epoch = 0
wait = 0
losses = []

# 학습 루프
print("Start training...")
stopped_early = False  # 종료 이유 기록 변수

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, dynamic_ncols=True)

    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
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
        best_epoch = epoch + 1  # ← 추가
        wait = 0
        torch.save(model.state_dict(), MODEL_PATH)
        tqdm.write(f"Model improved. Saved to {MODEL_PATH} (epoch {best_epoch}, loss={best_loss:.4f})")
    else:
        wait += 1
        if wait >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
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
plt.ylabel("MSE Loss")
plt.grid()
plt.savefig(LOSS_PLOT_PATH)
plt.close()

print("Training complete.")
print(f"Best model saved to: {MODEL_PATH}")
print(f"Loss plot saved to: {LOSS_PLOT_PATH}")
