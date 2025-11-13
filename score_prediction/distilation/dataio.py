# =========================================
# dataio.py — Unified (label_dir 기반, Fully Stabilized)
# =========================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------
# 1️⃣ Config & Device Utility
# ------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str="auto"):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ------------------------------------------------
# 2️⃣ Universal Table Loader
# ------------------------------------------------
def _read_any_table(path, excel_sheet=None, excel_header=0, select_numeric_only=True, fill_nan_value=0.0):
    """엑셀/CSV/NPY 파일을 자동 인식해 DataFrame으로 로드"""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=excel_sheet, header=excel_header)
        if isinstance(df, dict):  # 다중시트 대응
            df = list(df.values())[0]
    else:
        df = pd.read_csv(path, header=excel_header)

    # 숫자형 변환 시도
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # 선택적으로 숫자형만 유지
    if select_numeric_only and df.select_dtypes(include=["number"]).shape[1] > 0:
        df = df.select_dtypes(include=["number"])

    # 결측치 채움
    df = df.fillna(fill_nan_value)

    return df


# ------------------------------------------------
# 3️⃣ File Finder
# ------------------------------------------------
def _find_existing_file(base_dir, candidates):
    """후보 파일명 리스트 중 실제 존재하는 파일 경로 반환"""
    base_dir = Path(base_dir)
    for root, _, files in os.walk(base_dir):
        for name in candidates:
            for ext in [".csv", ".xlsx", ".xls", ".npy"]:
                if f"{name}{ext}" in files:
                    return str(Path(root) / f"{name}{ext}")
    raise FileNotFoundError(f"None of {candidates} found under {base_dir}")


# ------------------------------------------------
# 4️⃣ Info Printer
# ------------------------------------------------
def _print_feature_info(df, name):
    print(f"\n📊 [{name}] Feature Summary")
    print(f"  • 총 샘플 수: {len(df):,}")
    print(f"  • 피처 개수: {df.shape[1]}")
    print(f"  • 결측치 평균 개수: {df.isna().sum().mean():.1f}")
    print(f"  • 피처 목록: {list(df.columns[:8])}{' ...' if len(df.columns) > 8 else ''}")
    if len(df) > 0:
        print(f"  • 예시 1행: {df.iloc[0].to_dict()}")
    else:
        print("  • 예시 1행: N/A")


# ------------------------------------------------
# 5️⃣ Main Loader
# ------------------------------------------------
def load_inputs_and_labels(cfg):
    """A,B,Y 자동 탐색 및 전처리"""
    base_dir = Path(os.path.dirname(__file__)) / cfg["paths"]["data_dir"]
    label_dir = Path(cfg["paths"].get("label_dir", base_dir))  # ✅ label_dir만 사용
    fn_cfg = cfg["filenames"]
    excel_cfg = cfg.get("excel", {})
    prep_cfg = cfg.get("preprocess", {})

    # -------------------------------
    # 파일 탐색
    # -------------------------------
    A_path = _find_existing_file(base_dir, fn_cfg["A_candidates"])
    B_path = _find_existing_file(base_dir, fn_cfg["B_candidates"])
    Y_path = _find_existing_file(label_dir, fn_cfg["Y_candidates"])

    print("[load_inputs_and_labels] Found files:")
    print(f"  A → {os.path.basename(A_path)}")
    print(f"  B → {os.path.basename(B_path)}")
    print(f"  Y → {os.path.basename(Y_path)}")

    # -------------------------------
    # 파일 로드
    # -------------------------------
    A_df = _read_any_table(
        A_path,
        excel_sheet=excel_cfg.get("sheet_A"),
        excel_header=excel_cfg.get("header", 0),
        select_numeric_only=True,
        fill_nan_value=prep_cfg.get("fill_nan_value", 0.0),
    )

    B_df = _read_any_table(
        B_path,
        excel_sheet=excel_cfg.get("sheet_B"),
        excel_header=excel_cfg.get("header", 0),
        select_numeric_only=True,
        fill_nan_value=prep_cfg.get("fill_nan_value", 0.0),
    )

    Y_df = _read_any_table(
        Y_path,
        excel_sheet=excel_cfg.get("sheet_Y"),
        excel_header=excel_cfg.get("header", 0),
        select_numeric_only=False,  # ✅ Y는 반드시 전체 컬럼 유지
        fill_nan_value=prep_cfg.get("fill_nan_value", 0.0),
    )

    # -------------------------------
    # 라벨 필터링
    # -------------------------------
    expected_cols = ["attitude", "civic", "burden", "behavior"]
    valid_cols = [c for c in expected_cols if c in Y_df.columns]

    if len(valid_cols) == 0:
        print("⚠️ [WARNING] No valid Y columns found in label file.")
        print(f"    Available columns: {list(Y_df.columns)}")
        Y_df = pd.DataFrame(np.zeros((len(A_df), 1)), columns=["dummy"])
    else:
        dropped = [c for c in Y_df.columns if c not in valid_cols]
        if dropped:
            print(f"[WARN] Extra columns dropped from Y: {dropped}")
        Y_df = Y_df[valid_cols]

    # -------------------------------
    # 정보 출력
    # -------------------------------
    _print_feature_info(A_df, "A (Quantitative)")
    _print_feature_info(B_df, "B (Qualitative)")
    _print_feature_info(Y_df, "Y (Label)")

    # -------------------------------
    # NumPy 변환 + Scaling
    # -------------------------------
    A = StandardScaler().fit_transform(A_df.to_numpy(dtype=np.float32))
    B = StandardScaler().fit_transform(B_df.to_numpy(dtype=np.float32))
    Y = Y_df.to_numpy(dtype=np.float32)

    # -------------------------------
    # Shape 검증
    # -------------------------------
    if not (len(A) == len(B) == len(Y)):
        raise ValueError(f"❌ Size mismatch among A,B,Y → A={A.shape}, B={B.shape}, Y={Y.shape}")

    print(f"\n✅ [LOAD COMPLETE] Shapes →  A={A.shape}, B={B.shape}, Y={Y.shape}")
    return A, B, Y
