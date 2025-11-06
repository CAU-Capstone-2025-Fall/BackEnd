# ============================================================
# 🧠 Teacher–Student Representation Alignment & Explainability Suite (Refined + Income Normalization in SHAP)
# ============================================================
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# ⚙️ 폰트 설정 (한글 깨짐 방지)
# ============================================================
plt.rcParams["axes.unicode_minus"] = False
if os.name == "nt":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif os.name == "posix":
    if "AppleGothic" in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"

# ============================================================
# 0️⃣ 경로 및 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "results_analysis"
SAVE_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results_dist"

A_PATH = DATA_DIR / "A_processed.csv"
Z_TEACHER_PATH = RESULT_DIR / "Z_teacher.csv"
Z_STUDENT_PATH = RESULT_DIR / "Z_student.csv"
STUDENT_ENCODER_PATH = RESULT_DIR / "student_encoder.pt"

print("📂 Loading data...")
A = pd.read_csv(A_PATH)
Z_teacher = pd.read_csv(Z_TEACHER_PATH)
Z_student = pd.read_csv(Z_STUDENT_PATH)
assert A.shape[0] == Z_teacher.shape[0] == Z_student.shape[0], "❌ Row mismatch!"
print(f"✅ Loaded: A={A.shape}, Z_teacher={Z_teacher.shape}, Z_student={Z_student.shape}")

# ============================================================
# 1️⃣ PCA / t-SNE 시각화
# ============================================================
print("\n[STEP 1] Visualizing Teacher–Student Latent Alignment")

def visualize_latent_alignment(Z_teacher, Z_student, method="pca"):
    Z_t_np = Z_teacher.to_numpy()
    Z_s_np = Z_student.to_numpy()

    if method == "pca":
        reducer = PCA(n_components=2)
        Z_t_2d = reducer.fit_transform(Z_t_np)
        Z_s_2d = reducer.transform(Z_s_np)

    elif method == "tsne":
        # ✅ joint fit: teacher + student 합쳐서 같은 공간으로 매핑
        Z_all = np.concatenate([Z_t_np, Z_s_np])
        labels = np.array([0]*len(Z_t_np) + [1]*len(Z_s_np))
        reducer = TSNE(n_components=2, random_state=42)
        Z_all_2d = reducer.fit_transform(Z_all)
        Z_t_2d, Z_s_2d = Z_all_2d[:len(Z_t_np)], Z_all_2d[len(Z_t_np):]

    else:
        raise ValueError("method must be pca or tsne")

    plt.figure(figsize=(6,5))
    plt.scatter(Z_t_2d[:,0], Z_t_2d[:,1], c="#FF6B6B", alpha=0.5, label="Teacher")
    plt.scatter(Z_s_2d[:,0], Z_s_2d[:,1], c="#4D96FF", alpha=0.5, label="Student")
    plt.legend()
    plt.title(f"Teacher vs Student — {method.upper()} Projection")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"latent_alignment_{method}.png", dpi=200)
    plt.close()

for method in ["pca", "tsne"]:
    visualize_latent_alignment(Z_teacher, Z_student, method)
print("✅ Saved latent alignment visualizations (PCA/t-SNE).")

# ============================================================
# 2️⃣ Teacher–Student Correlation Heatmap
# ============================================================
print("\n[STEP 2] Computing correlation heatmap")

n_latent = Z_teacher.shape[1]
corr_matrix = np.zeros((n_latent, n_latent))
for i in range(n_latent):
    for j in range(n_latent):
        corr_matrix[i,j] = np.corrcoef(Z_teacher.iloc[:,i], Z_student.iloc[:,j])[0,1]

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            xticklabels=[f"Student_{i+1}" for i in range(n_latent)],
            yticklabels=[f"Teacher_{i+1}" for i in range(n_latent)])
plt.title("Teacher–Student Latent Correlation Matrix")
plt.tight_layout()
plt.savefig(SAVE_DIR / "latent_correlation_heatmap.png", dpi=200)
plt.close()
print("✅ Saved correlation heatmap.")

# ============================================================
# 3️⃣ Latent별 R² & Pearson
# ============================================================
print("\n[STEP 3] Computing latent-wise R² & Pearson correlation")

r2_each = r2_score(Z_teacher, Z_student, multioutput="raw_values")
pearson_each = [np.corrcoef(Z_teacher.iloc[:,i], Z_student.iloc[:,i])[0,1] for i in range(n_latent)]
df_corr = pd.DataFrame({
    "Latent": [f"latent_{i+1}" for i in range(n_latent)],
    "R²": r2_each,
    "Pearson_r": pearson_each
})
df_corr.to_csv(SAVE_DIR / "latent_metric_report.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(6,4))
sns.barplot(data=df_corr.melt(id_vars="Latent", var_name="Metric", value_name="Score"),
            x="Latent", y="Score", hue="Metric", palette="Blues")
plt.title("Latent-wise R² & Pearson_r")
plt.tight_layout()
plt.savefig(SAVE_DIR / "latent_metrics_barplot.png", dpi=200)
plt.close()
print("✅ Saved latent metric barplot.")

# ============================================================
# 4️⃣ SHAP Feature Attribution (Student Encoder)
# ============================================================
print("\n[STEP 4] SHAP Analysis using Actual Student Encoder")

device = "cuda" if torch.cuda.is_available() else "cpu"

class StudentEncoder(nn.Module):
    def __init__(self, input_dim=24, latent_dim=6):
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

# ✅ Load trained student encoder
state_dict = torch.load(STUDENT_ENCODER_PATH, map_location=device, weights_only=True)
student_encoder = StudentEncoder(input_dim=A.shape[1], latent_dim=n_latent)
student_encoder.load_state_dict(state_dict)
student_encoder.eval()

# ✅ Prepare DataFrame for SHAP
X_sample_df = A.iloc[:300].copy().astype(np.float32)

# 🔹 월평균 가구소득 + 연령 컬럼 정규화 (데이터는 그대로, SHAP 계산용만 변경)
scale_cols = [col for col in X_sample_df.columns if ("가구소득" in col or "연령" in col)]
if len(scale_cols) > 0:
    for col in scale_cols:
        vals = np.log1p(X_sample_df[col]) if X_sample_df[col].max() > 10 else X_sample_df[col]
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        X_sample_df[col] = vals
    print(f"✅ Normalized (for SHAP only): {scale_cols}")

# SHAP 결과 저장용
all_shap_summary = []

for i in range(n_latent):
    print(f"   → explaining student_latent_{i+1}")
    explainer = shap.Explainer(
        lambda x: student_encoder(torch.tensor(x.values, dtype=torch.float32).to(device))
                        .detach().cpu().numpy()[:, i],
        X_sample_df
    )
    shap_values = explainer(X_sample_df)

    # ✅ Summary Plot (한글 폰트 적용)
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=X_sample_df,
        feature_names=X_sample_df.columns,
        show=False
    )
    plt.title(f"Feature Influence on Student latent_{i+1}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"student_shap_latent{i+1}.png", dpi=200)
    plt.close()

    # ✅ SHAP values → DataFrame → CSV 저장
    df_shap = pd.DataFrame({
        "Feature": X_sample_df.columns,
        "MeanAbs_SHAP": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("MeanAbs_SHAP", ascending=False)
    df_shap.to_csv(SAVE_DIR / f"student_shap_latent{i+1}.csv", index=False, encoding="utf-8-sig")

    df_shap["Latent"] = f"latent_{i+1}"
    all_shap_summary.append(df_shap)

pd.concat(all_shap_summary, ignore_index=True).to_csv(
    SAVE_DIR / "student_shap_all_latents.csv", index=False, encoding="utf-8-sig"
)

print("✅ Saved SHAP plots and CSV tables (income normalized only in SHAP).")

# ============================================================
# 5️⃣ Downstream Predictive Check
# ============================================================
print("\n[STEP 5] Downstream Evaluation: Teacher vs Student Latent")

target = (A.iloc[:,0] > A.iloc[:,0].mean()).astype(int)
def downstream_eval(Z, label):
    X_train, X_test, y_train, y_test = train_test_split(Z, target, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return {"Model": label, "Accuracy": acc}

res = [downstream_eval(Z_teacher, "Teacher latent"),
       downstream_eval(Z_student, "Student latent")]

df_down = pd.DataFrame(res)
df_down.to_csv(SAVE_DIR / "downstream_accuracy.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(4,3))
sns.barplot(data=df_down, x="Model", y="Accuracy", palette="viridis")
plt.title("Downstream Accuracy: Teacher vs Student")
plt.tight_layout()
plt.savefig(SAVE_DIR / "downstream_accuracy.png", dpi=200)
plt.close()
print("✅ Saved downstream accuracy comparison.")

# ============================================================
# 6️⃣ Summary Report
# ============================================================
summary = {
    "Latent Alignment (PCA/tSNE)": "latent_alignment_[...].png",
    "Latent Correlation Heatmap": "latent_correlation_heatmap.png",
    "Latent R²/Pearson": "latent_metric_report.csv",
    "Feature Influence (SHAP)": "student_shap_latent*.png / .csv",
    "Downstream Accuracy": "downstream_accuracy.csv"
}
pd.DataFrame(summary.items(), columns=["Analysis Step", "Result File"]).to_csv(
    SAVE_DIR / "summary_report.csv", index=False, encoding="utf-8-sig"
)
print("\n✅ All analyses complete!")
print(f"📁 Results saved in: {SAVE_DIR}")
