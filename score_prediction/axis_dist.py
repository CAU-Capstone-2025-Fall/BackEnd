import numpy as np
import pandas as pd

# ============================================================ #
# 1️⃣ 데이터 로드
# ============================================================ #
A = pd.read_csv("./data/A_processed.csv")
B = pd.read_csv("./data/B_processed.csv")
Z_teacher = pd.read_csv("./results_dist/Z_teacher.csv")  # teacher latent (A+B 기반)

print(f"✅ Loaded A={A.shape}, B={B.shape}, Z_teacher={Z_teacher.shape}")

# ============================================================ #
# 2️⃣ 상관분석 함수
# ============================================================ #
def compute_top_corr(df_features, Z, prefix, top_k=5):
    corr_matrix = pd.DataFrame(index=df_features.columns, columns=Z.columns)
    for latent_col in Z.columns:
        for feat_col in df_features.columns:
            corr = np.corrcoef(df_features[feat_col], Z[latent_col])[0, 1]
            corr_matrix.loc[feat_col, latent_col] = corr
    corr_matrix = corr_matrix.astype(float)

    summary = []
    for latent in Z.columns:
        top_feats = corr_matrix[latent].abs().sort_values(ascending=False).head(top_k)
        for feat in top_feats.index:
            summary.append({
                "latent": latent,
                "feature": feat,
                "corr": corr_matrix.loc[feat, latent]
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"./results_dist/teacher_latent_topcorr_{prefix}.csv", index=False, encoding="utf-8-sig")
    return summary_df

# ============================================================ #
# 3️⃣ A, B, A+B 각각 실행
# ============================================================ #
summary_A  = compute_top_corr(A, Z_teacher, "A")
summary_B  = compute_top_corr(B, Z_teacher, "B")

AB = pd.concat([A, B], axis=1)
summary_AB = compute_top_corr(AB, Z_teacher, "AB")

# ============================================================ #
# 4️⃣ 결과 출력 (요약)
# ============================================================ #
print("\n📊 Teacher latent correlation analysis completed:")
print("   • ./results/teacher_latent_topcorr_A.csv")
print("   • ./results/teacher_latent_topcorr_B.csv")
print("   • ./results/teacher_latent_topcorr_AB.csv")

print("\n💡 Example Preview (Top 5 features from A+B):")
for i in range(1, min(7, Z_teacher.shape[1] + 1)):
    subset = summary_AB[summary_AB["latent"] == f"teacher_latent_{i}"]
    print(f"\n🧩 Latent_{i}:")
    for _, row in subset.iterrows():
        sign = "▲" if row["corr"] > 0 else "▼"
        print(f"   • {row['feature']:<40} | r = {row['corr']:.3f} {sign}")
