# ============================================================
# 🧠 Teacher Encoder SHAP Analysis (AutoEncoder 구조 호환 + 한글 폰트)
# ============================================================
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

# ============================================================
# ⚙️ 한글 폰트 설정
# ============================================================
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

if os.name == "nt":  # Windows
    plt.rcParams["font.family"] = "Malgun Gothic"
elif os.name == "posix":
    font_paths = fm.findSystemFonts(fontpaths=None, fontext="ttf")
    if any("AppleGothic" in p for p in font_paths):
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1️⃣ AutoEncoder 구조 (저장된 모델과 동일하게)
# ============================================================
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

# ============================================================
# 2️⃣ 모델 로드
# ============================================================
SAVE_DIR = Path(__file__).resolve().parent / "results_dist"
TEACHER_PATH = SAVE_DIR / "teacher_autoencoder.pt"

A = pd.read_csv("data/A_processed.csv")
B = pd.read_csv("data/B_processed.csv")
AB = pd.concat([A, B], axis=1)
print(f"✅ Loaded: A={A.shape}, B={B.shape}, A+B={AB.shape}")

# teacher 모델 전체 불러오기
teacher_model = AutoEncoder(input_dim=AB.shape[1], latent_dim=6)
teacher_model.load_state_dict(torch.load(TEACHER_PATH, map_location=device, weights_only=True))
teacher_model.eval()
teacher_encoder = teacher_model.encoder  # ✅ encoder만 사용

# ============================================================
# 3️⃣ SHAP 계산
# ============================================================
X_sample_df = AB.iloc[:300].copy().astype(np.float32)

# 🔹 log+minmax scale (income, age)
scale_cols = [c for c in X_sample_df.columns if ("가구소득" in c or "연령" in c)]
for col in scale_cols:
    vals = np.log1p(X_sample_df[col])
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    X_sample_df[col] = vals
print(f"✅ Normalized (for SHAP only): {scale_cols}")

# ============================================================
# 4️⃣ SHAP 수행
# ============================================================
all_shap_summary = []
n_latent = 6

for i in range(n_latent):
    print(f"→ explaining teacher_latent_{i+1}")

    explainer = shap.Explainer(
        lambda x: teacher_encoder(torch.tensor(x.values, dtype=torch.float32).to(device))
                        .detach().cpu().numpy()[:, i],
        X_sample_df
    )
    shap_values = explainer(X_sample_df)

    # ✅ Summary Plot (한글 지원)
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=X_sample_df,
        feature_names=X_sample_df.columns,
        show=False
    )
    plt.title(f"Teacher latent_{i+1}의 Feature 영향도")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"teacher_shap_latent{i+1}.png", dpi=200)
    plt.close()

    # ✅ SHAP CSV 저장
    df_shap = pd.DataFrame({
        "Feature": X_sample_df.columns,
        "MeanAbs_SHAP": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("MeanAbs_SHAP", ascending=False)
    df_shap["Latent"] = f"latent_{i+1}"
    df_shap.to_csv(SAVE_DIR / f"teacher_shap_latent{i+1}.csv", index=False, encoding="utf-8-sig")

    all_shap_summary.append(df_shap)

# 전체 통합 저장
pd.concat(all_shap_summary).to_csv(SAVE_DIR / "teacher_shap_all_latents.csv", index=False, encoding="utf-8-sig")

print("\n✅ Teacher SHAP analysis complete.")
print(f"📁 Results saved to: {SAVE_DIR}")
