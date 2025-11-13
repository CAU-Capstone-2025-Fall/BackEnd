# ============================================================
# 🐾 Label 생성 — 반려동물 사육경험 (1=없음, 2=과거, 3=현재)
# ============================================================

import re

import pandas as pd

# 1️⃣ 원본 엑셀 로드
df = pd.read_excel("survey.xlsx", sheet_name="마이크로데이터", header=[0, 1])
df.columns = df.columns.get_level_values(1)

# 2️⃣ 컬럼명 정리
clean_cols = {c: re.sub(r"\s+", "", str(c)) for c in df.columns}

target_col = None
for orig, clean in clean_cols.items():
    if "반려동물사육경험" in clean or "A1" in clean:
        target_col = orig
        break

if target_col is None:
    raise ValueError("⚠️ '반려동물 사육 경험(A1)' 항목을 찾을 수 없습니다.")

print(f"✅ 감지된 컬럼명: {repr(target_col)}")

# 3️⃣ 값 추출 및 정수 변환
exp_values = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")

# 4️⃣ 유효값만 남기기
valid_mask = exp_values.isin([1, 2, 3])
invalid_count = (~valid_mask).sum()
if invalid_count > 0:
    print(f"⚠️ 비유효 값 {invalid_count}개 → 제거됨")
exp_values = exp_values[valid_mask]

# 5️⃣ 분포 확인
print("📊 라벨 분포:")
print(exp_values.value_counts().sort_index())

# 6️⃣ CSV 저장
df_y = pd.DataFrame({"experience": exp_values})
df_y.to_csv("Y_experience.csv", index=False, encoding="utf-8-sig")

print("\n🎯 완료 — 저장됨: Y_experience.csv (클래스=3)")
