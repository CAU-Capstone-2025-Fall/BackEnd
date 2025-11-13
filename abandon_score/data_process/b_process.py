# ============================================================
# 🐶 B_processed_final_filtered.py
#   - survey.xlsx 로드
#   - 복수응답형(0/1), Likert(0~1) 정규화
#   - 반려동물 사육 경험 있음(A1=1,2) + 유기충동경험(B3>1) 필터링
# ============================================================

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1️⃣ 파일 로드
# ============================================================
file_path = "data/survey.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="마이크로데이터", header=[0, 1])
df_raw.columns = df_raw.columns.get_level_values(1)
df = df_raw.copy()

# ============================================================
# 2️⃣ 기본 정리: 불필요한 컬럼 제거
# ============================================================
drop_cols = [
    "ID", "SQ1", "SQ2", "SQ2_R", "SQ3_2", "SQ3_2_R","A1_1_1","A1_1_2","A1_1_3","A4",
    "DQ1", "DQ2", "DQ2_ETC4", "DQ3", "DQ4", "DQ5"
]
drop_keywords = ["기타", "ETC", "OPEN", "일련번호", "자치구", "권역"]

df = df.drop(columns=[c for c in df.columns if any(k in str(c) for k in drop_keywords) or c in drop_cols],
             errors="ignore")

# ============================================================
# 3️⃣ 필터: 반려동물 경험 + 유기충동 체크
# ============================================================
# A1 (반려동물 사육경험): 1=현재, 2=과거
df["A1"] = pd.to_numeric(df["A1"], errors="coerce")
mask_pet_experience = df["A1"].isin([1, 2])

# B3 (유기충동 경험): 1~3 Likert, 1=전혀없다 → 2~3 선택자 유지
df["B3"] = pd.to_numeric(df["B3"], errors="coerce")
mask_abandon_experience = df["B3"] >= 1

# 두 조건 모두 만족하는 사람만 남김
df_filtered = df[mask_pet_experience & mask_abandon_experience].copy()
print(f"✅ 필터링 완료: {len(df_filtered)}명 (전체 중 {len(df_filtered)/len(df):.1%}) 유지")

# ============================================================
# 4️⃣ 복수응답형 매핑
# ============================================================
multi_mapping = {
    "A2#": {
        1: "관리비용부담", 2: "이웃가족갈등", 3: "이상행동위생문제",
        4: "시간공간부족", 5: "여행외출곤란", 6: "죽음슬픔", 7: "가출실종",
    },
    "A3#": {
        1: "깨끗하게키울자신없음", 2: "주거환경나쁨", 3: "사육비용부담",
        4: "시간부족", 5: "공간없음", 6: "동물싫어함", 7: "가족반대",
    },
    "A5#": {
        1: "냄새심함", 2: "털날림", 3: "소음", 4: "대소변오염",
        5: "물리거나위협", 6: "교통사고", 7: "공원식당불편", 9: "피해없음",
    },
    "B1#": {
        1: "교육관리", 2: "예방치료", 3: "목욕운동",
        4: "좋은먹이", 5: "습성교육", 6: "공중규범",
    },
    "B2#": {
        1: "비용부담", 2: "가족갈등", 3: "위생문제", 4: "여건곤란", 5: "여행어려움",
    },
    "A1_2": {
        1: "아이들정서교육", 2: "예쁘고귀여워서", 3: "외로워서", 4: "우연히기회", 5: "유기견불쌍",
    },
}

def clean_to_codes(value):
    """'1, 3', '1 3', '1;3' 등을 [1,3]으로 변환"""
    if pd.isna(value):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    tokens = re.split(r"[ ,;/]+", str(value).strip())
    return [int(t) for t in tokens if t.isdigit()]

df_result = df_filtered.copy()

for prefix, mapping in multi_mapping.items():
    target_cols = [c for c in df_result.columns if c.startswith(prefix.replace("#", ""))]
    if not target_cols:
        continue

    for col in target_cols:
        df_result[col] = df_result[col].apply(clean_to_codes)
        for code, label in mapping.items():
            new_col = f"{prefix.replace('#', '')}_{label}"
            has_code = df_result[col].apply(lambda lst: code in lst)
            if new_col not in df_result.columns:
                df_result[new_col] = has_code.astype(int)
            else:
                df_result[new_col] = df_result[new_col] | has_code.astype(int)

    df_result.drop(columns=target_cols, inplace=True, errors="ignore")

print("✅ 복수응답형 변환 완료 (모든 항목 0/1 처리됨)")

# ============================================================
# 5️⃣ A1 원핫 인코딩
# ============================================================
if "A1" in df_result.columns:
    A1_map = {
        1: "현재_반려동물_있음",
        2: "과거에는_있었으나_지금은_없음",
        3: "반려동물_경험_없음",
    }
    for code, label in A1_map.items():
        df_result[f"A1_{label}"] = (df_result["A1"] == code).astype(int)
    df_result.drop(columns=["A1"], inplace=True)
print("✅ A1 원핫 인코딩 완료")

# ============================================================
# 6️⃣ A4 Likert 정규화
# ============================================================
if "A4" in df_result.columns:
    df_result["A4"] = pd.to_numeric(df_result["A4"], errors="coerce").fillna(0)
    scaler_A4 = MinMaxScaler()
    df_result["A4_norm"] = scaler_A4.fit_transform(df_result[["A4"]])
    df_result.drop(columns=["A4"], inplace=True)
print("✅ A4 정규화 완료 (0~1)")

# ============================================================
# 7️⃣ C5/C6 순위형 → 원핫
# ============================================================
C5_labels = {
    1: "기본소양교육", 2: "구조보호", 3: "예방및치료", 4: "훈련습성화",
    5: "사료용품구입", 6: "여행관리", 7: "소비자피해상담", 8: "장례시설", 10: "필요사업없음"
}

for col in ["C5_1", "C5_2", "C5_3"]:
    if col not in df_result.columns:
        continue
    df_result[col] = pd.to_numeric(df_result[col], errors="coerce")
    for code, label in C5_labels.items():
        if code == 9:  # 기타는 무시
            continue
        new_col = f"{col}_{label}"
        df_result[new_col] = (df_result[col] == code).astype(int)
    df_result.drop(columns=[col], inplace=True)

print("✅ C5/C6 순위형 원핫 인코딩 완료")

# ============================================================
# 8️⃣ Likert 0~1 정규화 문항
# ============================================================
likert_cols = [
    "C1","C2","C3_1","C3_2","C3_3","C3_4","C3_5","C3_6","C3_7","C3_8","C4","C6",
]
likert_names = [
    "동물보호센터 운영 인지정도","서울시의 폭넓은 동물보호센터 운영 찬성정도",
    "시민복지 관점 정부 관심 필요","자치구 기능만으로 정부 역할 부족","중앙정부 서울시 컨트롤타워 역할 필요",
    "공공 사육관리교육 갈등조정 필요","민간부문이 담당하기 어려운 영역 존재","복지시설투자보다 시민복지우선",
    "반려인 책임 강조 공공역할 최소","공공사업시 민간단체시설 활용","반려동물 관련 정부 중요 역할 인식","동물복지지원시설이 들어선다면 이에 대해 어떻게 생각하십니까?"
]
rename_dict = dict(zip(likert_cols, likert_names))
existing_cols = [c for c in likert_cols if c in df_result.columns]
df_result.rename(columns=rename_dict, inplace=True)
likert_cols = [rename_dict[c] for c in existing_cols]

df_result[likert_cols] = df_result[likert_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
scaler = MinMaxScaler()
df_result[likert_cols] = scaler.fit_transform(df_result[likert_cols])

print(f"✅ Likert 정규화 완료 ({len(likert_cols)}개)")

# ============================================================
# 9️⃣ 저장
# ============================================================
df_result = df_result.fillna(0)
output_path = "data/B_processed_filtered.csv"
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"🎯 저장 완료: {output_path}")
print("📘 예시 컬럼:", list(df_result.columns[:20]))
print(df_result[[c for c in df_result.columns if c.startswith("A2") or c.startswith("A3")]].head(10))
