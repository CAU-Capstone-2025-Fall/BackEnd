# ============================================================
# make_labels_from_B.py  (Revised: label source columns dropped)
# ============================================================
import numpy as np
import pandas as pd


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def make_labels_from_B(B_path, save_B_clean=True, save_path=None):
    """B 데이터에서 주요 변수만 남기고 4개 라벨 점수 생성 + label source 컬럼 제거"""
    df = pd.read_excel(B_path) if B_path.endswith((".xlsx", ".xls")) else pd.read_csv(B_path)
    print(f"[INFO] Loaded B data: shape={df.shape}")

    # 🔹 1. 주요 변수 선택 (label 생성에 사용될 컬럼들)
    label_cols = [
        "반려동물 유기 충동 경험", "새로운 반려동물 사육 의향",
        "서울시의 폭넓은 동물보호센터 운영 찬성정도", "시민복지 관점 정부 관심 필요",
        "중앙정부 서울시 컨트롤타워 역할 필요", "반려인 책임 강조 공공역할 최소",
        "A2_관리비용부담", "A2_이웃가족갈등", "A3_시간부족", "A5_털날림",
        "B1_교육관리", "B1_예방치료", "C6_훈련습성화"
    ]
    exist_cols = [c for c in label_cols if c in df.columns]
    print(f"[INFO] Label-related columns found: {len(exist_cols)} / {len(label_cols)}")

    # 🔹 2. 결측치 처리
    df = df.fillna(df.mean())

    # 🔹 3. 라벨 계산
    attitude = zscore(df["새로운 반려동물 사육 의향"]) - zscore(df["반려동물 유기 충동 경험"])
    civic = (
        zscore(df["서울시의 폭넓은 동물보호센터 운영 찬성정도"])
        + zscore(df["시민복지 관점 정부 관심 필요"])
        + zscore(df["중앙정부 서울시 컨트롤타워 역할 필요"])
        - zscore(df["반려인 책임 강조 공공역할 최소"])
    ) / 4
    burden = -zscore(df[["A2_관리비용부담", "A2_이웃가족갈등", "A3_시간부족", "A5_털날림"]]).mean(axis=1)
    behavior = zscore(df[["B1_교육관리", "B1_예방치료", "C6_훈련습성화"]]).mean(axis=1)

    # 🔹 4. 라벨 DataFrame
    label_df = pd.DataFrame({
        "attitude": attitude,
        "civic": civic,
        "burden": burden,
        "behavior": behavior,
    })
    label_df = (label_df - label_df.min()) / (label_df.max() - label_df.min())

    print("[INFO] Label summary:")
    print(label_df.describe().round(3))

    # 🔹 5. 라벨로 사용된 컬럼 제거 → 학습용 B_clean 생성
    B_clean = df.drop(columns=[c for c in label_cols if c in df.columns])
    print(f"[INFO] Cleaned B shape: {B_clean.shape}")

    # 🔹 6. 저장
    if save_path:
        label_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved labels → {save_path}")

    if save_B_clean:
        clean_path = B_path.replace(".csv", "_clean.csv").replace(".xlsx", "_clean.csv")
        B_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved cleaned B → {clean_path}")

    return B_clean, label_df


if __name__ == "__main__":
    B_clean, label_df = make_labels_from_B(
        B_path="data/B.csv",
        save_path="data/Y_4labels.csv"
    )
