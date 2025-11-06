# ============================================================ #
# 🧠 Teacher–Student Distillation Pipeline (CPU-only, Local)
# ============================================================ #
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================================================ #
# 1️⃣ 데이터 로드 (상대경로 ./data/)
# ============================================================ #
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
A_path = os.path.join(DATA_DIR, "A_processed.csv")
B_path = os.path.join(DATA_DIR, "B_processed.csv")

A = pd.read_csv(A_path)
B = pd.read_csv(B_path)
print(f"✅ Loaded data: A={A.shape}, B={B.shape}")

# ============================================================ #
# 2️⃣ 정규화
# ============================================================ #
scaler = MinMaxScaler()
A = pd.DataFrame(scaler.fit_transform(A), columns=A.columns)
B = pd.DataFrame(scaler.fit_transform(B), columns=B.columns)
AB = pd.concat([A, B], axis=1)

# ============================================================ #
# 3️⃣ AutoEncoder 정의 (Teacher)
# ============================================================ #
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=6):
        super().__init__()
        h1, h2 = max(128, input_dim // 2), max(64, input_dim // 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.1),
            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.LeakyReLU(0.1),
            nn.Linear(h2, h1),
            nn.LeakyReLU(0.1),
            nn.Linear(h1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ============================================================ #
# 4️⃣ AutoEncoder 학습 (Teacher)
# ============================================================ #
def train_autoencoder(X, latent_dim=6, epochs=150, lr=1e-4):
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=64, shuffle=True)
    model = AutoEncoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs), desc="Training Teacher AutoEncoder"):
        for xb, _ in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        _, Z = model(X_tensor)
    return model, Z.numpy()

teacher_model, Z_teacher = train_autoencoder(AB, latent_dim=6, epochs=150)
print(f"✅ Teacher latent shape: {Z_teacher.shape}")

# ============================================================ #
# 5️⃣ Student 정의 (A-only)
# ============================================================ #
class StudentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        return self.model(x)

# ============================================================ #
# 6️⃣ Student Distillation 학습
# ============================================================ #
A_tensor = torch.tensor(A.values, dtype=torch.float32)
Z_teacher_tensor = torch.tensor(Z_teacher, dtype=torch.float32)
dataset = TensorDataset(A_tensor, Z_teacher_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

student = StudentEncoder(input_dim=A.shape[1], latent_dim=6)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)

def distill_loss(z_s, z_t, alpha=0.7):
    mse = torch.mean((z_s - z_t) ** 2)
    cosine = 1 - torch.nn.functional.cosine_similarity(z_s, z_t).mean()
    return alpha * mse + (1 - alpha) * cosine

for epoch in tqdm(range(100), desc="Training Student (Distillation)"):
    total_loss = 0
    for xb, zt in loader:
        optimizer.zero_grad()
        zs = student(xb)
        loss = distill_loss(zs, zt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss={total_loss/len(loader):.6f}")

# ============================================================ #
# 7️⃣ Latent 일관성 검증
# ============================================================ #
student.eval()
with torch.no_grad():
    Z_student = student(A_tensor).numpy()

cos_sim = np.mean(np.sum(Z_teacher * Z_student, axis=1) /
                  (np.linalg.norm(Z_teacher, axis=1) * np.linalg.norm(Z_student, axis=1)))
print(f"💡 Avg Cosine Similarity (Teacher vs Student): {cos_sim:.3f}")

# 시각화 (선택적)
Z_all = np.concatenate([Z_teacher, Z_student])
y_label = np.array([0]*len(Z_teacher) + [1]*len(Z_student))
Z_emb = TSNE(n_components=2, random_state=42).fit_transform(Z_all)

plt.figure(figsize=(7,6))
plt.scatter(Z_emb[y_label==0,0], Z_emb[y_label==0,1], alpha=0.6, label="Teacher latent")
plt.scatter(Z_emb[y_label==1,0], Z_emb[y_label==1,1], alpha=0.6, label="Student latent")
plt.legend()
plt.title("Latent Space Alignment (Teacher vs Student)")
plt.show()

# ============================================================ #
# 8️⃣ 모델 및 결과 저장
# ============================================================ #
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

torch.save(teacher_model.state_dict(), os.path.join(RESULT_DIR, "teacher_autoencoder.pt"))
torch.save(student.state_dict(), os.path.join(RESULT_DIR, "student_encoder.pt"))
pd.DataFrame(Z_teacher, columns=[f"teacher_latent_{i+1}" for i in range(6)]).to_csv(
    os.path.join(RESULT_DIR, "Z_teacher.csv"), index=False)
pd.DataFrame(Z_student, columns=[f"student_latent_{i+1}" for i in range(6)]).to_csv(
    os.path.join(RESULT_DIR, "Z_student.csv"), index=False)

print("\n✅ Saved models and latents in ./results/")
