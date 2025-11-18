import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("data/survey.xlsx", sheet_name="마이크로데이터", header=[0,1])
df.columns = df.columns.get_level_values(1)

# 1) 필터링 (사육경험자 + 유기충동 응답자)
df["A1"] = pd.to_numeric(df["A1"], errors="coerce")
df["B3"] = pd.to_numeric(df["B3"], errors="coerce")
mask = df["A1"].isin([1,2]) & df["B3"].notna()
df_target = df[mask].copy()

print("🐾 사육경험자:", df["A1"].isin([1,2]).sum())
print("📌 유기충동 응답자:", mask.sum())

# 2) A4 문자열 클린징 + 숫자 변환
def clean_num(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().replace("\n", "").replace("\t", "").replace(" ", "")

df_target["A4_clean"] = df_target["A4"].apply(clean_num)
df_target["A4_num"] = pd.to_numeric(df_target["A4_clean"], errors="coerce")

print("\n🔥 A4 원본 → 숫자 변환 unique:")
print(df_target["A4_num"].unique())

# 3) 값 추출
vals = df_target["A4_num"].values.reshape(-1,1)

# 4) -1~1 정규화
if len(np.unique(vals)) == 1:
    print("⚠ 값이 모두 동일 → 0으로 저장")
    norm_vals = np.zeros(len(vals))
else:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_vals = scaler.fit_transform(vals).flatten()
print(df["A4"].value_counts(dropna=False))
print(df.loc[df["A1"].isin([1,2]) & df["B3"].notna(), ["A1","B3","A4"]].head(20))

# 5) 저장
out = pd.DataFrame({"향후 반려동물 사육의향_norm": norm_vals})
out.to_csv("data/A4_labeled_norm.csv", index=False, encoding="utf-8-sig")
print(df.columns.tolist())

print("\n🎯 완료 — 저장됨: data/A4_labeled_norm.csv")
