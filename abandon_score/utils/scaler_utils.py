# ============================================================
# scaler_utils.py
#   - 저장된 StandardScaler 로드 및 적용
# ============================================================

import joblib
import pandas as pd

NUMERIC_COLS = ["연령", "가족 구성원 수", "주택규모", "월평균 가구소득"]

def transform_numeric_features(df, scaler_path="models/scaler_age_income.pkl"):
    """저장된 scaler를 불러와서 동일한 정규화 적용"""
    scaler = joblib.load(scaler_path)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df
