# ============================================================
# 🧩 Teacher vs MLP vs Student Distillation 성능 비교
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1️⃣ 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEACHER_PATH = BASE_DIR / "results_dist" / "Z_teacher.csv"          # Teacher 정답
MLP_PATH = BASE_DIR / "results_mlp" / "Z_pred_mlp.csv"              # 기존 MLP 예측 (없으면 주석 처리)
STUDENT_PATH = BASE_DIR / "results_dist" / "Z_student.csv"          # 새 distillation 예측

print(f"📂 Loading data from:")
print(f" - Teacher: {TEACHER_PATH}")
print(f" - MLP:     {MLP_PATH}")
print(f" - Student: {STUDENT_PATH}")

# ------------------------------------------------------------
# 2️⃣ 데이터 로드
# ------------------------------------------------------------
Y_teacher = pd.read_csv(TEACHER_PATH).values.astype(np.float32)
Y_mlp = pd.read_csv(MLP_PATH).values.astype(np.float32)
Y_student = pd.read_csv(STUDENT_PATH).values.astype(np.float32)

assert Y_teacher.shape == Y_mlp.shape == Y_student.shape, \
    f"❌ Shape mismatch: teacher={Y_teacher.shape}, mlp={Y_mlp.shape}, student={Y_student.shape}"

output_dim = Y_teacher.shape[1]

# ------------------------------------------------------------
# 3️⃣ 성능 계산 함수
# ------------------------------------------------------------
def evaluate(Y_true, Y_pred, name):
    mse = mean_squared_error(Y_true, Y_pred)
    mae = mean_absolute_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_true, Y_pred)
    r2_each = r2_score(Y_true, Y_pred, multioutput='raw_values')
    corr_each = [pearsonr(Y_true[:, i], Y_pred[:, i])[0] for i in range(output_dim)]
    df = pd.DataFrame({
        "Latent": [f"latent_{i+1}" for i in range(output_dim)],
        "R²": r2_each,
        "Pearson_r": corr_each
    })
    summary = pd.DataFrame({
        "Model": [name],
        "MSE": [mse],
        "RMSE": [rmse],
        "MAE": [mae],
        "R² (overall)": [r2]
    })
    return summary, df

# ------------------------------------------------------------
# 4️⃣ 평가 수행
# ------------------------------------------------------------
summary_mlp, detail_mlp = evaluate(Y_teacher, Y_mlp, "MLP")
summary_student, detail_student = evaluate(Y_teacher, Y_student, "Student")

# ------------------------------------------------------------
# 5️⃣ 결과 출력
# ------------------------------------------------------------
print("\n📊 전체 성능 비교 (전체 평균):")
summary_all = pd.concat([summary_mlp, summary_student], ignore_index=True)
print(summary_all.round(4))

print("\n📈 Student latent별 R² 및 Pearson 상관:")
print(detail_student.round(4))

# ------------------------------------------------------------
# 6️⃣ 저장
# ------------------------------------------------------------
SAVE_DIR = BASE_DIR / "results_compare"
SAVE_DIR.mkdir(exist_ok=True)
summary_all.to_csv(SAVE_DIR / "model_summary.csv", index=False, encoding="utf-8-sig")
detail_student.to_csv(SAVE_DIR / "student_detail.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ Saved comparison results to {SAVE_DIR}")
