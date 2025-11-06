# ============================================
# 🧩 teacher_latent_topcorr_AB 시각화 코드 (한글 폰트 지원 버전)
# ============================================

import platform
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# ✅ 한글 폰트 설정
system = platform.system()
if system == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows: 맑은 고딕
elif system == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux (예: Colab, Ubuntu)
    if "NanumGothic" in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams["font.family"] = "NanumGothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# ✅ 현재 실행 파일 기준 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results_dist" / "teacher_latent_topcorr_AB.csv"
SAVE_DIR = BASE_DIR / "results_dist" / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Loading CSV from: {CSV_PATH}")

# ✅ 파일 존재 확인
if not CSV_PATH.exists():
    raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {CSV_PATH}")

# ✅ 데이터 로드
teacher_latent_topcorrAB = pd.read_csv(CSV_PATH)
teacher_latent_topcorrAB["latent"] = teacher_latent_topcorrAB["latent"].astype(str)

print(f"✅ Loaded data shape: {teacher_latent_topcorrAB.shape}")

# ✅ 고유 latent별 시각화
for latent, group in teacher_latent_topcorrAB.groupby("latent"):
    plt.figure(figsize=(8, 4))

    # 절댓값 기준 정렬 (큰 값이 위로 오도록)
    group = group.reindex(group["corr"].abs().sort_values(ascending=True).index)

    # 색상 설정: 양수(빨강) / 음수(파랑)
    colors = ["#FF6B6B" if c > 0 else "#4D96FF" for c in group["corr"]]

    # 수평 막대그래프
    plt.barh(group["feature"], group["corr"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"{latent} — Feature Correlations", fontsize=13, pad=10)
    plt.xlabel("상관계수 (r)")
    plt.ylabel("특성 (Feature)")
    plt.tight_layout()

    # ✅ 저장 및 표시
    save_path = SAVE_DIR / f"{latent}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"📊 Saved plot: {save_path}")
