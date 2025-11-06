# ============================================================
# 🧠 Generate MLP prediction file (Z_pred_mlp.csv) + Debug Info
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "A_processed.csv"           # ✅ MLP 입력은 A
TEACHER_PATH = BASE_DIR / "results_dist" / "Z_teacher.csv"  # ✅ 정답 latent
MODEL_PATH = BASE_DIR / "results_mlp" / "final_mlp.pt"      # ✅ 저장된 MLP checkpoint
SAVE_PATH = BASE_DIR / "results_mlp" / "Z_pred_mlp.csv"     # ✅ 저장 위치

print("📂 Debug check for row count consistency")
print(f"A_processed.csv   → {DATA_PATH}")
print(f"Z_teacher.csv     → {TEACHER_PATH}")
print(f"final_mlp.pt      → {MODEL_PATH}")

# ------------------------------------------------------------
# 🔍 1️⃣ 파일별 행 개수 비교
# ------------------------------------------------------------
df_A = pd.read_csv(DATA_PATH)
df_teacher = pd.read_csv(TEACHER_PATH)

print(f"👉 A_processed.csv shape: {df_A.shape}")
print(f"👉 Z_teacher.csv shape:  {df_teacher.shape}")

if len(df_A) != len(df_teacher):
    print(f"⚠️ WARNING: Row count mismatch → A={len(df_A)} vs Teacher={len(df_teacher)}")
    diff = abs(len(df_A) - len(df_teacher))
    print(f"⚠️ Difference: {diff} rows")
else:
    print("✅ Row counts match exactly!")

# ------------------------------------------------------------
# 모델 구조 (학습 당시와 동일)
# ------------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------
X = df_A.values.astype(np.float32)
input_dim = X.shape[1]
output_dim = df_teacher.shape[1] if df_teacher.shape[1] else 6

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLPRegressor(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------------------------------------------
# 예측
# ------------------------------------------------------------
with torch.no_grad():
    X_tensor = torch.tensor(X, device=device)
    Y_pred = model(X_tensor).cpu().numpy()

print(f"✅ MLP predicted shape: {Y_pred.shape}")

# ------------------------------------------------------------
# 🔍 2️⃣ 행 개수 일치 여부 다시 확인
# ------------------------------------------------------------
if len(Y_pred) != len(df_teacher):
    print(f"⚠️ WARNING: Prediction count mismatch — MLP={len(Y_pred)} vs Teacher={len(df_teacher)}")
    min_len = min(len(Y_pred), len(df_teacher))
    print(f"⚙️ Trimming both to {min_len} samples for alignment.")
    Y_pred = Y_pred[:min_len]
    df_teacher = df_teacher.iloc[:min_len]

# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
# ------------------------------------------------------------
# 저장
# ------------------------------------------------------------
# ❌ 기존 (문제 발생)
# np.savetxt(SAVE_PATH, Y_pred, delimiter=",")

# ✅ 수정된 버전
pd.DataFrame(Y_pred, columns=[f"latent_{i+1}" for i in range(Y_pred.shape[1])]) \
  .to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Saved MLP prediction safely to {SAVE_PATH}")
