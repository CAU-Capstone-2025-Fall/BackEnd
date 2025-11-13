# ============================================================
# make_unlabeled_testset.py
#   - 반려동물 키운 경험은 있지만,
#     유기 충동 설문(B3)에 응답하지 않은 사람만 추출
# ============================================================

import re

import pandas as pd


# ------------------------------------------------------------
# 1️⃣ 컬럼명 탐색 함수
# ------------------------------------------------------------
def find_col_by_keywords(columns, keywords):
    for col in columns:
        clean = re.sub(r"\s+", "", str(col))
        if any(kw in clean for kw in keywords):
            return col
    return None


# ------------------------------------------------------------
# 2️⃣ 엑셀 로드
# ------------------------------------------------------------
try:
    df = pd.read_excel("data/survey.xlsx", sheet_name="마이크로데이터", header=[0, 1])
    df.columns = df.columns.get_level_values(1)
except Exception as e:
    raise FileNotFoundError(f"⚠️ survey.xlsx 로드 실패: {e}")

# ------------------------------------------------------------
# 3️⃣ 주요 컬럼 자동 감지
# ------------------------------------------------------------
col_A1 = find_col_by_keywords(df.columns, ["반려동물사육경험", "A1"])
col_B3 = find_col_by_keywords(df.columns, ["유기충동", "B3"])

if col_A1 is None or col_B3 is None:
    raise ValueError("⚠️ 'A1(사육경험)' 또는 'B3(유기충동)' 컬럼을 찾을 수 없습니다.")

print(f"✅ A1(사육경험): {col_A1}")
print(f"✅ B3(유기충동): {col_B3}")

# ------------------------------------------------------------
# 4️⃣ 반려동물 사육 경험자 필터
# ------------------------------------------------------------
A1_vals = pd.to_numeric(df[col_A1], errors="coerce")
mask_owner = A1_vals.isin([1, 2])
df_owner = df[mask_owner].copy()

# ------------------------------------------------------------
# 5️⃣ 유기 충동 미응답자만 선택
# ------------------------------------------------------------
B3_vals = pd.to_numeric(df_owner[col_B3], errors="coerce")

mask_invalid_b3 = B3_vals.isna() | (~B3_vals.isin([1, 2, 3]))
df_unlabeled = df_owner[mask_invalid_b3].copy()

n_total = len(df)
n_owner = len(df_owner)
n_unlabeled = len(df_unlabeled)

print(f"\n[INFO] 전체 응답자: {n_total}")
print(f"       ├─ 반려동물 사육 경험자: {n_owner}")
print(f"       └─ 그 중 '유기 충동 미응답자' (테스트셋): {n_unlabeled}")

# ------------------------------------------------------------
# 6️⃣ 결과 저장 (원본 컬럼 그대로)
# ------------------------------------------------------------
df_unlabeled.to_excel("data/survey_unlabeled_testset.xlsx", index=False)
print(f"\n📁 저장 완료 → data/survey_unlabeled_testset.xlsx (행 {len(df_unlabeled)})")
print("🧩 주의: 이 데이터는 label이 없으므로 A,B feature 생성 시 참조용으로만 사용하세요.")
