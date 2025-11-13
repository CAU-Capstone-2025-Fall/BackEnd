# ============================================================
# 🐶 A_processed 복원 — 경험/의향 관련 항목 제거
# ============================================================

import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ 엑셀 원본 로드
df = pd.read_excel("survey.xlsx", sheet_name="마이크로데이터", header=[0, 1])
df.columns = df.columns.get_level_values(1)

# 2️⃣ A_processed 불러오기
A = pd.read_csv("A.csv")

# 3️⃣ 제외 대상 컬럼 탐색 (사육 경험 + 사육 의향 관련)
exclude_keywords = ["반려동물사육의향", "반려동물사육경험", "A3", "A4"]
drop_cols = [
    c for c in A.columns
    if any(k in re.sub(r"\s+", "", str(c)) for k in exclude_keywords)
]

print(f"🧹 제거할 컬럼 {len(drop_cols)}개: {drop_cols}")

# 4️⃣ 제거 후 확인
A_fixed = A.drop(columns=drop_cols, errors="ignore")
print(f"✅ 최종 컬럼 수: {A_fixed.shape[1]}")

# 5️⃣ 저장
output_path = "A_processed_clean.csv"
A_fixed.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n🎯 완료 — 저장됨: {output_path}")
