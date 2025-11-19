# routers/inference.py
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from lime.lime_tabular import LimeTabularExplainer
from model.models import StudentNet

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

sys.path.append(BASE_DIR)

# ------------------------------------------------------------
# Load global probability distribution for percentile
# ------------------------------------------------------------
ALL_PROBS_PATH = os.path.join(DATA_DIR, "all_probs.npy")

if os.path.exists(ALL_PROBS_PATH):
    ALL_PROBS = np.load(ALL_PROBS_PATH).reshape(-1)
    print(f"[INF] Loaded all_probs.npy: {ALL_PROBS.shape}")
else:
    ALL_PROBS = None
    print("[WARN] all_probs.npy not found. Percentile disabled.")


# ------------------------------------------------------------
# UX-friendly probability smoothing (중간대 눌러버리는 방식)
# ------------------------------------------------------------
def adjust_probability(p):
    """
    UX-friendly smoothing.
    0~0.3: 더 낮추기
    0.3~0.7: 중간대 압축
    0.7~1.0: 더 높여 강조
    """
    if p < 0.3:
        return p * 0.8
    elif p < 0.7:
        return 0.3 + (p - 0.3) * 0.4
    else:
        return min(1.0, 0.7 + (p - 0.7) * 1.3)


# ------------------------------------------------------------
# Percentile 계산
# ------------------------------------------------------------
def compute_percentile(prob):
    if ALL_PROBS is None or len(ALL_PROBS) == 0:
        return None

    rank = np.sum(ALL_PROBS < prob)
    percentile = 100 - int((rank / len(ALL_PROBS)) * 100)

    return percentile


# ------------------------------------------------------------
# 1. Load scaler
# ------------------------------------------------------------
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_age_income.pkl")
SCALER = joblib.load(SCALER_PATH)
print("[INF] Loaded scaler:", SCALER_PATH)


# ------------------------------------------------------------
# 2. Load config.yaml
# ------------------------------------------------------------
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

STU_CFG = CONFIG["student_model"]
TASK_CFG = CONFIG["task"]


# ------------------------------------------------------------
# 3. Initialize StudentNet
# ------------------------------------------------------------
DIM_A = 16
DIM_Y = TASK_CFG["dim_y"]
NUM_CLASSES = TASK_CFG["num_classes"]

DEVICE = "cpu"

student = StudentNet(
    dim_A=DIM_A,
    dim_y=DIM_Y,
    num_classes=NUM_CLASSES,
    z_dim=STU_CFG["z_dim"],
    enc_hidden=STU_CFG["enc_hidden"],
    clf_hidden=STU_CFG["clf_hidden"],
    p_drop=STU_CFG["p_drop"],
    use_layernorm=STU_CFG["use_layernorm"]
).to(DEVICE)
student.eval()

# ------------------------------------------------------------
# 4. Load checkpoint
# ------------------------------------------------------------
CKPT_PATH = os.path.join(CKPT_DIR, CONFIG["student_train"]["ckpt_name"])
state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
student.load_state_dict(state_dict)
print("[INF] StudentNet loaded:", CKPT_PATH)


# ------------------------------------------------------------
# 5. Define predict_proba (for LIME)
# ------------------------------------------------------------
def student_predict_proba(X_numpy):
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_logits, _, _ = student(X_tensor)
        probs = torch.softmax(y_logits, dim=-1).cpu().numpy()
    return probs


# ------------------------------------------------------------
# 6. Load A.csv → LIME background
# ------------------------------------------------------------
FEATURE_NAMES = [
    "연령", "가족 구성원 수", "주택규모", "월평균 가구소득",
    "성별_1", "성별_2",
    "주택형태_1", "주택형태_2", "주택형태_3", "주택형태_4",
    "향후 반려동물 사육의향",
    "화이트칼라", "블루칼라", "자영업", "비경제활동층", "기타"
]

scale_cols = ["연령", "가족 구성원 수", "주택규모", "월평균 가구소득"]

df_A = pd.read_csv(os.path.join(DATA_DIR, "A.csv"))
df_A_scaled = df_A.copy()
df_A_scaled[scale_cols] = SCALER.transform(df_A[scale_cols])

background = df_A_scaled.values[:200]  # only 200 samples
print(f"[INF] LIME background loaded: {background.shape}")


# ------------------------------------------------------------
# 7. LIME Explainer
# ------------------------------------------------------------
lime_explainer = LimeTabularExplainer(
    training_data=background,
    feature_names=FEATURE_NAMES,
    class_names=["0", "1"],
    mode="classification",
    discretize_continuous=True
)


# ------------------------------------------------------------
# 8. LIME → clean feature name + mapping
# ------------------------------------------------------------
def clean_feature_name(raw):
    tokens = raw.split()
    for t in tokens:
        if re.match(r'^[가-힣A-Za-z_]', t):
            return t.strip()
    return raw.strip()


HUMAN_MAP = {
    "연령": "연령",
    "가족 구성원 수": "가족 구성원 수",
    "주택규모": "주택 규모",
    "월평균 가구소득": "월평균 가구소득",
    "향후 반려동물 사육의향": "사육 의향",

    "성별_1": "남성",
    "성별_2": "여성",

    "주택형태_1": "아파트",
    "주택형태_2": "단독/다가구",
    "주택형태_3": "연립/빌라/다세대",
    "주택형태_4": "기타 주거형태",

    "화이트칼라": "화이트칼라",
    "블루칼라": "블루칼라",
    "자영업": "자영업",
    "비경제활동층": "비경제활동층",
    "기타": "기타 직업군",
}

def human_name(n):
    return HUMAN_MAP.get(n, n)


def infer_lime(df_scaled):
    x = df_scaled.values[0]
    explanation = lime_explainer.explain_instance(
        data_row=x,
        predict_fn=student_predict_proba,
        num_features=16
    )

    raw_list = explanation.as_list()
    result = {}

    for raw_key, weight in raw_list:
        clean = clean_feature_name(raw_key)
        pretty = human_name(clean)
        result[pretty] = float(weight)

    sorted_items = sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
    return dict(sorted_items)


# ------------------------------------------------------------
# 9. Main inference
# ------------------------------------------------------------
def infer_student(features: dict):

    A_COLUMNS = [
        "연령", "가족 구성원 수", "주택규모", "월평균 가구소득",
        "성별_1", "성별_2",
        "주택형태_1", "주택형태_2", "주택형태_3", "주택형태_4",
        "향후 반려동물 사육의향",
        "화이트칼라", "블루칼라", "자영업", "비경제활동층", "기타"
    ]

    df_raw = pd.DataFrame([features])

    df_scaled_part = pd.DataFrame(
        SCALER.transform(df_raw[scale_cols]),
        columns=scale_cols
    )

    df_rest = df_raw[[c for c in A_COLUMNS if c not in scale_cols]]
    df_final = pd.concat([df_scaled_part, df_rest], axis=1)[A_COLUMNS]

    x = torch.tensor(df_final.values, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_logits, z_a, feats = student(x)

    probs = torch.softmax(y_logits, dim=-1).cpu().numpy()[0]
    prob_raw = float(probs[1])            # 모델 원본 확률
    prob_adj = adjust_probability(prob_raw)  # 🔥 UX-friendly 확률

    percentile = compute_percentile(prob_adj)

    return {
        "input_raw": df_raw.to_dict("records")[0],
        "input_scaled": df_final.to_dict("records")[0],
        "latent_vector": z_a.cpu().numpy().tolist()[0],
        "logits": y_logits.cpu().numpy().tolist()[0],
        "probability": prob_adj,       # 🔥 사용자 표시 확률
        "percentile": percentile,      
        "features": feats
    }
