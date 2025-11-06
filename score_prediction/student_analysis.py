# ============================================================
# 🧩 student_analysis.py
# Teacher–Student Representation Alignment & Explainability Suite
# ============================================================

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# 0️⃣ 경로 및 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "results_analysis"
SAVE_DIR.mkdir(exist_ok=True)

A_PATH = BASE_DIR / "data" / "A_processed.csv"
Z_TEACHER_PATH = BASE_DIR / "results_dist" / "Z_teacher.csv"
Z_STUDENT_PATH = BASE_DIR / "results_dist" / "Z_student.csv"

print("📂 Loading data...")
A = pd.read_csv(A_PATH)
Z_teacher = pd.read_csv(Z_TEACHER_PATH)
Z_student = pd.read_csv(Z_STUDENT_PATH)
assert A.shape[0] == Z_teacher.shape[0] == Z_student.shape[0], "❌ Row mismatch!"

print(f"✅ Loaded: A={A.shape}, Z_teacher={Z_teacher.shape}, Z_student={Z_student.shape}")

# ============================================================
# 1️⃣ PCA / t-SNE / UMAP 시각화
# ============================================================
print("\n[STEP 1] Visualizing Teacher–Student Latent Alignment")

def visualize_latent_alignment(Z_teacher, Z_student, method="pca"):
    # numpy 변환 (feature name 충돌 방지)
    Z_t_np = Z_teacher.to_numpy()
    Z_s_np = Z_student.to_numpy()

    if method == "pca":
        reducer = PCA(n_components=2)
        Z_t_2d = reducer.fit_transform(Z_t_np)
        Z_s_2d = reducer.transform(Z_s_np)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        Z_t_2d = reducer.fit_transform(Z_t_np)
        Z_s_2d = reducer.fit_transform(Z_s_np)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z_t_2d = reducer.fit_transform(Z_t_np)
        Z_s_2d = reducer.fit_transform(Z_s_np)
    else:
        raise ValueError("method must be pca, tsne, or umap")

    plt.figure(figsize=(6, 5))
    plt.scatter(Z_t_2d[:, 0], Z_t_2d[:, 1], alpha=0.5, c="#FF6B6B", label="Teacher")
    plt.scatter(Z_s_2d[:, 0], Z_s_2d[:, 1], alpha=0.5, c="#4D96FF", label="Student")
    plt.legend()
    plt.title(f"Teacher vs Student — {method.upper()} Projection")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"latent_alignment_{method}.png", dpi=200)
    plt.close()

for method in ["pca", "tsne", "umap"]:
    visualize_latent_alignment(Z_teacher, Z_student, method)
print("✅ Saved latent alignment visualizations (PCA/t-SNE/UMAP).")

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
# 4️⃣ SHAP-based Feature Influence Comparison (Teacher vs Student)
# ============================================================
print("\n[STEP 4] SHAP Comparison: Teacher vs Student Latent Attribution")

# SHAP을 위해 각 latent에 대해 간단히 회귀 근사
# (Teacher latent, Student latent 각각을 입력 A로부터 근사)
import xgboost as xgb

X_sample = A.values[:300]
feature_names = A.columns

for i in range(n_latent):
    print(f"   → Explaining latent_{i+1}")
    y_teacher = Z_teacher.iloc[:,i]
    y_student = Z_student.iloc[:,i]

    # XGBoost 회귀로 근사 후 SHAP 해석
    model_teacher = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_student = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_teacher.fit(A, y_teacher)
    model_student.fit(A, y_student)

    expl_teacher = shap.Explainer(model_teacher, A)
    expl_student = shap.Explainer(model_student, A)
    shap_t = expl_teacher(X_sample)
    shap_s = expl_student(X_sample)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    shap.summary_plot(shap_t, A, feature_names=feature_names, show=False)
    plt.title(f"Teacher latent_{i+1}")

    plt.subplot(1,2,2)
    shap.summary_plot(shap_s, A, feature_names=feature_names, show=False)
    plt.title(f"Student latent_{i+1}")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"shap_compare_latent{i+1}.png", dpi=200)
    plt.close()

print("✅ Saved SHAP comparison plots (Teacher vs Student).")

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
    "Latent Alignment (PCA/tSNE/UMAP)": "latent_alignment_[...].png",
    "Latent Correlation Heatmap": "latent_correlation_heatmap.png",
    "Latent R²/Pearson": "latent_metric_report.csv",
    "Feature Influence (SHAP)": "shap_compare_latent*.png",
    "Downstream Accuracy": "downstream_accuracy.csv"
}
pd.DataFrame(summary.items(), columns=["Analysis Step", "Result File"]).to_csv(
    SAVE_DIR / "summary_report.csv", index=False, encoding="utf-8-sig"
)
print("\n✅ All analyses complete!")
print(f"📁 Results saved in: {SAVE_DIR}")
