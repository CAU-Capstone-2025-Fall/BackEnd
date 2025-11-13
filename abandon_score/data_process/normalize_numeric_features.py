# ============================================================
# 🧮 normalize_numeric_features.py (수정판)
# ============================================================

import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1️⃣ 데이터 로드
# ------------------------------------------------------------
input_path = "data/A_labeled_merged.csv"
df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} samples from {input_path}")

# ------------------------------------------------------------
# 2️⃣ 정규화할 수치형 컬럼 지정
# ------------------------------------------------------------
NUMERIC_COLS = ["연령", "가족 구성원 수", "주택규모", "월평균 가구소득"]

# ------------------------------------------------------------
# 3️⃣ StandardScaler 정규화
# ------------------------------------------------------------
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# ------------------------------------------------------------
# 4️⃣ 경로 및 디렉터리 생성
# ------------------------------------------------------------
scaled_path = "data/A_labeled_normalized.csv"
scaler_path = "models/scaler_age_income.pkl"

os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # <── 추가!

# ------------------------------------------------------------
# 5️⃣ 저장
# ------------------------------------------------------------
df.to_csv(scaled_path, index=False, encoding="utf-8-sig")
joblib.dump(scaler, scaler_path)

print("\n🎯 완료 — 저장됨:")
print(f" - 정규화된 데이터: {scaled_path}")
print(f" - 스케일러 객체: {scaler_path}")
