import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (없으면 깨짐)
# 시스템에 맞는 폰트 경로를 지정해야 합니다.
try:
    # Windows
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
except FileNotFoundError:
    try:
        # macOS
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    except FileNotFoundError:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다 (글자가 깨질 수 있음).")
        # Linux 등 다른 환경에서는 맞는 폰트 경로를 설정해야 합니다.

# 모델 파일 로드
MODEL_PATH = "ltr_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"모델 '{MODEL_PATH}' 로드 성공.")
except FileNotFoundError:
    print(f"모델 파일을 찾을 수 없습니다. '{MODEL_PATH}'가 현재 폴더에 있는지 확인하세요.")
    exit()

# (매우 중요) 학습 때 사용한 피처 이름 목록이 반드시 필요합니다!
# 순서와 내용이 train_ltr_model.py와 정확히 일치해야 합니다.
FEATURE_COLUMNS = [
    'sim_mix', 'compat_score', 'priority_score', 'location_score',
    'allergy_penalty', 'conflict_penalty', 'sim_x_compat'
]

# 모델에 피처 이름 설정
model.booster_.feature_name_ = FEATURE_COLUMNS

# 1. 피처 중요도 시각화 (gain 기준)
# gain: 해당 피처를 사용함으로써 얻는 정보 이득의 총량 (가장 중요한 지표)
fig, ax = plt.subplots(figsize=(10, 8))
lgb.plot_importance(model, ax=ax, importance_type='gain', title='Feature Importance (Gain)')
plt.tight_layout()
plt.savefig("feature_importance_gain.png")
print("피처 중요도(gain) 그래프를 'feature_importance_gain.png' 파일로 저장했습니다.")

# 2. 개별 결정 트리 시각화 (모델의 내부 로직 확인)
# 모델을 구성하는 수많은 트리 중 첫 번째 트리를 그려봅니다.
# graphviz 라이브러리가 필요합니다: pip install graphviz
try:
    ax_tree = lgb.plot_tree(model, tree_index=0, figsize=(20, 16), show_info=['split_gain'])
    plt.tight_layout()
    plt.savefig("decision_tree_0.png")
    print("첫 번째 결정 트리 그래프를 'decision_tree_0.png' 파일로 저장했습니다.")
except ImportError:
    print("\n[알림] 'graphviz' 라이브러리가 없어 결정 트리 시각화를 건너뜁니다.")
    print("트리 시각화를 원하시면 'pip install graphviz'와 시스템에 Graphviz를 설치해주세요.")

# 모든 그래프를 화면에 표시
plt.show()