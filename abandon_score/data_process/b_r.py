# ============================================================
# 🐶 B_grouped_scores_v6.py
#   - B_processed_filtered.csv 로드
#   - C5 계열: one-hot 기반 순위 근사 가중치 적용
#   - 나머지 그룹은 평균 정규화
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1️⃣ 데이터 로드
# ------------------------------------------------------------
input_path = "data/B_processed_filtered.csv"
df = pd.read_csv(input_path)
print(f"✅ 로드 완료: {df.shape}")

# ------------------------------------------------------------
# 2️⃣ C5 항목(one-hot 기반) 가중치 계산
# ------------------------------------------------------------
weights = {"C5_1_": 1.0, "C5_2_": 0.66, "C5_3_": 0.33}

# C5 관련 컬럼만 필터링
C5_cols = [c for c in df.columns if c.startswith("C5_")]
C5_scores = pd.DataFrame(index=df.index)

for label_prefix in ["기본소양교육","구조보호","예방및치료","훈련습성화",
                     "사료용품구입","여행관리","소비자피해상담","장례시설","필요사업없음"]:
    score = np.zeros(len(df))
    for prefix, w in weights.items():
        col = f"{prefix}{label_prefix}"
        if col in df.columns:
            score += df[col].astype(float) * w
    C5_scores[label_prefix] = score

# 정규화
scaler = MinMaxScaler()
C5_scores = pd.DataFrame(
    np.round(scaler.fit_transform(C5_scores), 3),
    columns=C5_scores.columns,
    index=df.index
)
print(f"✅ C5 순위 기반(근사) 인코딩 완료: {C5_scores.shape}")

# ------------------------------------------------------------
# 3️⃣ 나머지 그룹 매핑
# ------------------------------------------------------------
group_map = {
    "위생_민감성": ["A5_냄새심함","A5_털날림","A5_소음","A5_대소변오염"],
    "안전_우려": ["A5_물리거나위협","A5_교통사고"],
    "사회적_불편감": ["A5_공원식당불편","B2_가족갈등","B2_위생문제"],
    "피해_없음": ["A5_피해없음"],
    "돌봄_실행력": ["B1_교육관리","B1_예방치료","B1_목욕운동","B1_좋은먹이"],
    "윤리_규범의식": ["B1_습성교육","B1_공중규범"],
    "경제적_부담": ["B2_비용부담"],
    "시간_공간_제약": ["B2_여건곤란","B2_여행어려움"],
    "정서적_애정표현형": ["A1_2_예쁘고귀여워서"],
    "정서적_공감형": ["A1_2_아이들정서교육","A1_2_유기견불쌍"],
    "정서적_의존형": ["A1_2_외로워서","A1_2_우연히기회"],
    "자율_책임균형": ["B1_교육관리","B1_공중규범","B2_비용부담","B2_여건곤란"]
}

group_scores = pd.DataFrame(index=df.index)

for group_name, cols in group_map.items():
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        continue
    raw = df[valid_cols].mean(axis=1)
    scaled = MinMaxScaler().fit_transform(raw.values.reshape(-1, 1)).flatten()
    group_scores[group_name] = np.round(scaled, 3)

# ------------------------------------------------------------
# 4️⃣ C5 하위 4개 그룹 통합
# ------------------------------------------------------------
C5_subgroups = {
    "공공서비스_교육훈련형": ["기본소양교육","훈련습성화"],
    "공공서비스_보호의료형": ["구조보호","예방및치료"],
    "공공서비스_생활편의형": ["사료용품구입","여행관리"],
    "공공서비스_제도행정형": ["소비자피해상담","장례시설","필요사업없음"]
}

for sub_name, cols in C5_subgroups.items():
    valid_cols = [c for c in cols if c in C5_scores.columns]
    if not valid_cols:
        continue
    raw = C5_scores[valid_cols].mean(axis=1)
    scaled = MinMaxScaler().fit_transform(raw.values.reshape(-1, 1)).flatten()
    group_scores[sub_name] = np.round(scaled, 3)

# ------------------------------------------------------------
# 5️⃣ 저장
# ------------------------------------------------------------
df_out = pd.concat([df, group_scores], axis=1)
output_path = "data/B_grouped_scores_v6.csv"
df_out.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.3f")

print(f"🎯 저장 완료: {output_path}")
print(group_scores.head(10))
