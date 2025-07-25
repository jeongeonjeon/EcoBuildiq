import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# 하이퍼파라미터 설정
window_size = 24
batch_size = 64
epochs = 20
learning_rate = 0.001

# .npy 파일에서 입력 시퀀스와 타깃 불러오기
X = np.load("data/processed/X_lstm.npy").astype(np.float32)
y = np.load("data/processed/y_lstm.npy").astype(np.float32)

# 학습/검증 데이터셋 분할 (80% 학습, 20% 검증)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# PyTorch 텐서로 변환하고 DataLoader 구성
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력값 사용
        return out.squeeze()

# 모델, 손실 함수, 최적화 도구 정의
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 손실 및 검증 손실 저장 리스트
train_losses, val_losses = [], []

# 학습 반복
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(-1)  # 입력 형태 맞추기
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.unsqueeze(-1)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    # 평균 손실 계산
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # 학습 상태 출력
    print("Epoch {}/{} - Train Loss: {:.4f} - Val Loss: {:.4f}".format(
        epoch+1, epochs, train_loss, val_loss))

# 모델 저장
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/lstm_model.pth")

# 손실 시각화
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss")
plt.legend()
plt.savefig("models/lstm_loss_curve.png")
# plt.show()
