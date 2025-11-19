# te.py (offline_precompute_probs.py)

import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------
# 1) backend 루트를 Python 경로에 포함
# ------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
sys.path.append(BASE_DIR)

# ------------------------------------
# 2) 경로 추가 이후에 import 해야 됨
# ------------------------------------
from routers.inference import infer_student

# ------------------------------------
# 3) 데이터 로드
# ------------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "data/A.csv"))

probs = []

print("총 샘플:", len(df))

# ------------------------------------
# 4) 전체 확률 예측
# ------------------------------------
for i in range(len(df)):
    row = df.iloc[i].to_dict()
    res = infer_student(row)
    probs.append(res["probability"])

# ------------------------------------
# 5) 저장
# ------------------------------------
save_path = os.path.join(BASE_DIR, "data/all_probs.npy")
np.save(save_path, np.array(probs))

print("저장 완료:", save_path, "개수 =", len(probs))
