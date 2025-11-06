# ============================================================
# 🐶 A_processed 복원 — '향후 반려동물 사육의향' (1~5 → 0~1)
# ============================================================

import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ 엑셀 원본 로드
df = pd.read_excel("survey.xlsx", sheet_name="마이크로데이터", header=[0,1])
df.columns = df.columns.get_level_values(1)

# 2️⃣ 컬럼명에서 줄바꿈 제거 후 일치 탐색
clean_cols = {c: re.sub(r"\s+", "", str(c)) for c in df.columns}
target_col = None
for orig, clean in clean_cols.items():
    if "향후반려동물사육의향" in clean or "A4" in clean:
        target_col = orig
        break

if target_col is None:
    raise ValueError("⚠️ '향후 반려동물 사육 의향' 문항을 찾을 수 없습니다.")

print(f"✅ 감지된 컬럼명: {repr(target_col)}")

# 3️⃣ 값 추출 (1~5 Likert)
A4_values = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values.reshape(-1, 1)
print("📊 원본 분포:")
print(pd.Series(A4_values.flatten()).value_counts().sort_index())

# 4️⃣ 0~1 정규화
scaler = MinMaxScaler()
A4_norm = scaler.fit_transform(A4_values)

# 5️⃣ A_processed 불러오기 및 대체
A = pd.read_csv("A_processed.csv")
A["향후 반려동물 사육의향"] = A4_norm

# 6️⃣ 저장
output_path = "A_processed_fixed.csv"
A.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n🎯 완료 — 저장됨: {output_path}")
print(A[["향후 반려동물 사육의향"]].describe())
