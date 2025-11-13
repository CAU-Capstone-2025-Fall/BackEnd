import os

import numpy as np
import pandas as pd


def _load_csv_or_npy(path):
    """CSV 또는 NPY 파일을 로드"""
    if path.endswith(".csv"):
        return pd.read_csv(path).to_numpy()
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {path}")

def load_inputs_and_labels(cfg):
    """config.yaml을 기반으로 A, B, Y 데이터를 로드"""
    data_dir = cfg['paths']['data_dir']
    label_dir = cfg['paths']['label_dir']
    
    a_path = os.path.join(data_dir, cfg['filenames']['A_name'])
    b_path = os.path.join(data_dir, cfg['filenames']['B_name'])
    y_path = os.path.join(label_dir, cfg['filenames']['Y_name'])
    
    if not os.path.exists(a_path): raise FileNotFoundError(f"A 파일 없음: {a_path}")
    if not os.path.exists(b_path): raise FileNotFoundError(f"B 파일 없음: {b_path}")
    if not os.path.exists(y_path): raise FileNotFoundError(f"Y 파일 없음: {y_path}")

    print(f"[DataIO] Loading A from: {a_path}")
    A = _load_csv_or_npy(a_path)
    
    print(f"[DataIO] Loading B from: {b_path}")
    B = _load_csv_or_npy(b_path)
    
    print(f"[DataIO] Loading Y from: {y_path}")
    # Y는 (N, 1) 또는 (N,) 형태의 1D 벡터여야 함
    Y = _load_csv_or_npy(y_path)
    if Y.ndim > 1:
        print(f"[DataIO] Y shape {Y.shape} -> squeezing to 1D.")
        Y = Y.squeeze()

    # Y는 분류용이므로 정수(int) 타입이어야 함
    Y = Y.astype(np.int64)
    
    assert len(A) == len(B) == len(Y), "A, B, Y의 샘플 수가 일치하지 않습니다."
    
    print(f"✅ [DataIO] Load complete. Shapes: A={A.shape}, B={B.shape}, Y={Y.shape}")
    print(f"✅ [DataIO] Y unique values: {np.unique(Y)}")
    return A, B, Y