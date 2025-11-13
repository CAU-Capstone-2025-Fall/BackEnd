# ============================================================
# make_Y_impulse_binary.py
#   - Y = 유기 충동 경험 (이진 라벨)
#   - A1 = 반려동물 사육 경험 (1=현재, 2=과거)
# ============================================================

import re

import pandas as pd


# ------------------------------------------------------------
# 1️⃣ 컬럼명 탐색 함수
# ------------------------------------------------------------
def find_col_by_keywords(columns, keywords):
    """컬럼명 중 키워드가 포함된 첫 번째 컬럼 반환"""
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
# 3️⃣ 컬럼명 자동 탐지
# ------------------------------------------------------------
col_A1 = find_col_by_keywords(df.columns, ["반려동물사육경험", "A1"])
col_B3 = find_col_by_keywords(df.columns, ["유기충동", "B3"])

if col_A1 is None or col_B3 is None:
    raise ValueError("⚠️ '사육경험(A1)' 또는 '유기충동(B3)' 컬럼을 찾을 수 없습니다.")

print(f"✅ A1(사육경험) 컬럼: {col_A1}")
print(f"✅ B3(유기충동경험) 컬럼: {col_B3}")

# ------------------------------------------------------------
# 4️⃣ 반려동물 사육 경험자 필터링
# ------------------------------------------------------------
A1_vals = pd.to_numeric(df[col_A1], errors="coerce")
mask_owner = A1_vals.isin([1, 2])
df_owner = df[mask_owner].copy()

n_total = len(df)
n_owner = len(df_owner)
print(f"\n[INFO] 전체 응답자 {n_total}명 중 반려동물 사육 경험자 {n_owner}명")

# ------------------------------------------------------------
# 5️⃣ 유기 충동 응답 필터링
# ------------------------------------------------------------
B3_vals = pd.to_numeric(df_owner[col_B3], errors="coerce")

mask_valid_b3 = B3_vals.isin([1, 2, 3])
mask_invalid_b3 = ~mask_valid_b3 | B3_vals.isna()

n_with_impulse = mask_valid_b3.sum()
n_missing_impulse = mask_invalid_b3.sum()

print(f"  ├─ 그중 '유기 충동 설문에 응답한 사람' {n_with_impulse}명")
print(f"  └─ '유기 충동 응답이 비어있거나 비정상' {n_missing_impulse}명 (제외됨)")

# ------------------------------------------------------------
# 6️⃣ 유효 응답자만 남기고 라벨 생성 (이진)
# ------------------------------------------------------------
df_valid = df_owner[mask_valid_b3].copy()
Y_raw = B3_vals[mask_valid_b3].astype(int) - 1  # (1,2,3) → (0,1,2)
Y_bin = (Y_raw > 0).astype(int)  # 0 → 0, 1/2 → 1

# ------------------------------------------------------------
# 7️⃣ CSV 저장
# ------------------------------------------------------------
df_y = pd.DataFrame({
    "impulse": Y_raw.values,
    "impulse_binary": Y_bin.values
})
df_y.to_csv("data/label/Y_binary.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 8️⃣ 분포 요약 출력
# ------------------------------------------------------------
print("\n📊 '유기 충동 경험 (원래)' 분포 (0=없음, 1=가끔, 2=자주):")
print(df_y["impulse"].value_counts().sort_index())

print("\n📊 '유기 충동 경험 (이진)' 분포 (0=없음, 1=있음):")
print(df_y["impulse_binary"].value_counts().sort_index())

print(f"\n🎯 최종 Y 생성 완료 — 저장됨: data/label/Y_binary.csv (n={len(df_y)})")
print("🚨 A, B 데이터도 df_valid.index 기준으로 필터링해야 순서가 일치합니다.")
