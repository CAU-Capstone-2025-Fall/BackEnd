# ============================================================
# evaluate_regression.py (Y_4labels 회귀용 최종 평가)
# ============================================================

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml

# ⚠️ 'models.py'와 'train_utils.py'가 임포트 가능해야 함
from models import MLP, StudentNet, TeacherNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, random_split

# from train_utils import metrics_dict # (metrics_dict 대신 regression_metrics 사용)


# ============================================================
# 1️⃣ Validation 복원 + Label 매칭 (Regression)
# ============================================================
def load_holdout_data(A, B, label_path, val_split=0.2, seed=42):
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)

    # ❗️ [수정] label_path가 Path 객체일 수 있으므로 str() 처리
    label_path_str = str(label_path)
    if label_path_str.endswith(".csv"):
        df_label = pd.read_csv(label_path_str)
    else:
        df_label = pd.read_excel(label_path_str)
    
    # Y_4labels.csv의 4개 컬럼 (index 제외)
    feature_cols = [c for c in df_label.columns if c not in ["Unnamed: 0", "index"]]
    print(f"[INFO] Using label columns: {feature_cols}")

    assert len(A_t) == len(df_label), "A/B와 라벨 데이터의 샘플 수가 다릅니다."
    
    # ❗️ [수정] Y_t(레이블)를 'float' 타입으로 로드 (회귀)
    Y_t = torch.tensor(df_label[feature_cols].values, dtype=torch.float32)
    ds = TensorDataset(A_t, B_t, Y_t)

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_ds, va_ds = random_split(ds, [n_train, n_val], generator=gen)

    # 훈련 셋 (A_tr, B_tr, Y_tr)
    A_tr = torch.stack([s[0] for s in tr_ds])
    B_tr = torch.stack([s[1] for s in tr_ds])
    Y_tr = torch.stack([s[2] for s in tr_ds]) # (float 타입)

    # 검증 셋 (A_val, B_val, Y_val)
    A_val = torch.stack([s[0] for s in va_ds])
    B_val = torch.stack([s[1] for s in va_ds])
    Y_val = torch.stack([s[2] for s in va_ds]) # (float 타입)

    print(f"[LOAD] Train set: {len(A_tr)}, Validation set: {len(A_val)}")
    return A_tr, B_tr, Y_tr, A_val, B_val, Y_val, feature_cols


# ============================================================
# 2️⃣ Baseline (A only) (Regression)
# ============================================================
class BaselineNet(nn.Module):
    # ❗️ [수정] '분류'용 num_classes 인자 제거
    def __init__(self, dim_A, dim_y, enc_hidden, clf_hidden, z_dim, 
                 p_drop=0.1, use_layernorm=False):
        super().__init__()
        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)
        # ❗️ [수정] '분류기'가 아닌 '회귀(Regressor)'
        self.regressor = MLP(z_dim, dim_y, clf_hidden, p_drop, use_layernorm)

    def forward(self, x):
        z = self.encoder(x)
        y = self.regressor(z) # (Batch, 4) 예측값 반환
        return y, z


# ============================================================
# 3️⃣ Metrics (Regression)
# ============================================================
def regression_metrics(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    # ❗️ [수정] R2는 multi-output='variance_weighted' (표준) 또는 'uniform_average'
    r2 = r2_score(y_true_np, y_pred_np, multioutput='variance_weighted') 
    
    # 4개 레이블 각각의 R2도 계산
    r2_per_dim = r2_score(y_true_np, y_pred_np, multioutput='raw_values')
    
    metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}
    # 4개 특성 R2 추가 (예: R2_attitude)
    for i, col in enumerate(FEATURE_COLS):
        metrics[f"R2_{col}"] = r2_per_dim[i]
        
    return metrics


# ============================================================
# 4️⃣ 모델 평가 루틴 (Regression)
# ============================================================
def evaluate_models(
    A_tr, B_tr, Y_tr,
    A_val, B_val, Y_val,
    TeacherNet, StudentNet,
    teacher_ckpt, student_ckpt,
    teacher_model_cfg, student_model_cfg,
    device='cuda', out_dir="./eval_results"
):
    os.makedirs(out_dir, exist_ok=True)
    A_tr, Y_tr = A_tr.to(device), Y_tr.to(device, dtype=torch.float32)
    A_val, B_val, Y_val = A_val.to(device), B_val.to(device), Y_val.to(device, dtype=torch.float32)
    
    mse_loss = nn.MSELoss() # ❗️ Baseline 훈련용

    # --- 1. Teacher
    with torch.no_grad():
        # ❗️ [수정] num_classes 제거 (회귀 모델 __init__ 가정)
        teacher = TeacherNet(
            dim_A=teacher_model_cfg["dim_A"],
            dim_B=teacher_model_cfg["dim_B"],
            dim_y=teacher_model_cfg["dim_y"],
            encA_hidden=teacher_model_cfg["encA_hidden"],
            encB_hidden=teacher_model_cfg["encB_hidden"],
            clf_hidden=teacher_model_cfg["clf_hidden"],
            z_dim_A=teacher_model_cfg["z_dim_A"],
            z_dim_B=teacher_model_cfg["z_dim_B"],
            p_drop=teacher_model_cfg["p_drop"],
            use_layernorm=teacher_model_cfg["use_layernorm"]
        ).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        Yp_teacher, _, _ = teacher(A_val, B_val) # (B, 4)
        metrics_teacher = regression_metrics(Y_val, Yp_teacher)

    # --- 2. Student
    with torch.no_grad():
        student = StudentNet(
            dim_A=student_model_cfg["dim_A"],
            dim_y=student_model_cfg["dim_y"],
            enc_hidden=student_model_cfg["enc_hidden"],
            clf_hidden=student_model_cfg["clf_hidden"],
            z_dim=student_model_cfg["z_dim"],
            p_drop=student_model_cfg["p_drop"],
            use_layernorm=student_model_cfg["use_layernorm"]
        ).to(device)
        student.load_state_dict(torch.load(student_ckpt, map_location=device, weights_only=True))
        student.eval()
        Yp_student, _ = student(A_val) # (B, 4)
        metrics_student = regression_metrics(Y_val, Yp_student)

    # --- 3. Baseline
    baseline = BaselineNet(
        dim_A=student_model_cfg["dim_A"],
        dim_y=student_model_cfg["dim_y"],
        enc_hidden=student_model_cfg["enc_hidden"],
        clf_hidden=student_model_cfg["clf_hidden"],
        z_dim=student_model_cfg["z_dim"],
        p_drop=student_model_cfg["p_drop"],
        use_layernorm=student_model_cfg["use_layernorm"]
    ).to(device)

    opt = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 50 

    for ep in range(epochs):
        baseline.train()
        opt.zero_grad()
        y_pred, _ = baseline(A_tr)
        loss = mse_loss(y_pred, Y_tr) # ❗️ [수정] 회귀용 MSELoss
        loss.backward()
        opt.step()

    with torch.no_grad():
        baseline.eval()
        Yp_base, _ = baseline(A_val) # (B, 4)
        metrics_base = regression_metrics(Y_val, Yp_base)

    # --- 4. 결과 요약
    df = pd.DataFrame([
        {"Model": "Teacher (A+B)", **metrics_teacher},
        {"Model": "Student (Distilled)", **metrics_student},
        {"Model": "Baseline (A only)", **metrics_base},
    ])
    df_rounded = df.round(4)
    df.to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)
    print("\n===== 📊 Evaluation Summary =====")
    print(df_rounded)

    # --- 5. 시각화 (회귀용)
    # ❗️ [수정] R2, MSE 등 회귀 지표
    metrics_to_plot = ["R2", "MSE", "MAE"]
    palette = ["#ff7f0e", "#1f77b4", "#2ca02c"]
    n_metrics = len(metrics_to_plot)

    plt.figure(figsize=(max(5, n_metrics * 3), 5))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, n_metrics, i + 1)
        sns.barplot(x="Model", y=metric, hue="Model", data=df, palette=palette, legend=False, order=df['Model'])
        plt.title(metric)
        plt.xticks(rotation=25, ha='right')
        if metric == "R2":
            plt.axhline(0, color='black', linestyle='--', lw=1) # R2=0 기준선
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Model Performance on Hold-out Validation Set")
    plt.savefig(os.path.join(out_dir, "eval_barplots.png"), dpi=200)
    plt.close()

    # ❗️ [삭제] Scatter Plot은 4D라서 mean() 비교는 무의미
    
    # --- 6. 예측값 저장
    df_pred_true = pd.DataFrame(Y_val.cpu().numpy(), columns=FEATURE_COLS)
    df_pred_teacher = pd.DataFrame(Yp_teacher.cpu().numpy(), columns=[f"pred_T_{c}" for c in FEATURE_COLS])
    df_pred_student = pd.DataFrame(Yp_student.cpu().numpy(), columns=[f"pred_S_{c}" for c in FEATURE_COLS])
    df_pred_baseline = pd.DataFrame(Yp_base.cpu().numpy(), columns=[f"pred_B_{c}" for c in FEATURE_COLS])
    
    df_pred_all = pd.concat([df_pred_true, df_pred_teacher, df_pred_student, df_pred_baseline], axis=1)
    df_pred_all.to_csv(os.path.join(out_dir, "val_predictions_regression.csv"), index=False)
    
    print(f"[SAVE] All results saved → {out_dir}")

    return df


# ============================================================
# 5️⃣ 실행 예시
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "data"
    CKPT_DIR = BASE_DIR / "checkpoints"
    OUT_DIR = BASE_DIR / "eval_results"

    CONFIG_PATH = BASE_DIR / "config.yaml"
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print("[LOAD] config.yaml loaded.")

    # --- [수정] A_noex, B_clean 로드
    try:
        A = pd.read_csv(DATA_DIR / "A_noex.csv").to_numpy()
        B = pd.read_csv(DATA_DIR / "B_clean.csv").to_numpy()
    except FileNotFoundError:
        print(f"A_noex.csv/B_clean.csv not found, trying .npy")
        A = np.load(DATA_DIR / "A.npy")
        B = np.load(DATA_DIR / "B_clean.csv.npy") # B_clean.npy?

    # --- [수정] Y_4labels.csv 로드
    label_path = DATA_DIR / "label" / "Y_4labels.csv"
    if not label_path.exists():
        # Fallback (Y_candidates에서 찾기)
        print(f"Warning: {label_path} not found. Searching in config.filenames.Y_candidates...")
        Y_candidates = cfg['filenames']['Y_candidates']
        search_dirs = [DATA_DIR, DATA_DIR / "label", BASE_DIR.parent]
        for sdir in search_dirs:
            for cand in Y_candidates:
                p = sdir / (cand + ".csv")
                if p.exists(): label_path = p; break
            if label_path: break
        if not label_path:
            raise FileNotFoundError(f"Could not find Y_4labels.csv or candidates in {search_dirs}")

    print(f"[LOAD] Using Label file: {label_path}")

    # ❗️ [수정] FEATURE_COLS를 전역 변수로 설정하기 위해 로드
    A_tr, B_tr, Y_tr, A_val, B_val, Y_val, loaded_feature_cols = load_holdout_data(
        A, B, label_path, 
        val_split=cfg['teacher_train']['val_split'], 
        seed=cfg['seed']
    )
    FEATURE_COLS = loaded_feature_cols # 전역 변수 업데이트
    true_dim_y = Y_val.shape[1] # 4

    # --- config 딕셔너리에 dim 정보 주입
    # ❗️ [수정] 'num_classes' 키 제거 (회귀)
    cfg['teacher_model'].update({
        "dim_A": A.shape[1],
        "dim_B": B.shape[1],
        "dim_y": true_dim_y,
    })
    cfg['student_model'].update({
        "dim_A": A.shape[1],
        "dim_y": true_dim_y,
    })
    cfg['teacher_model'].pop('num_classes', None)
    cfg['student_model'].pop('num_classes', None)
    
    # (use_layernorm 동기화)
    if 'use_layernorm' not in cfg['teacher_model']:
         cfg['teacher_model']['use_layernorm'] = False
    if 'use_layernorm' not in cfg['student_model']:
         cfg['student_model']['use_layernorm'] = False

    # --- [수정] 체크포인트 이름
    teacher_ckpt = CKPT_DIR / "teacher.pt" # ❗️ (Optuna/config와 일치하는지 확인)
    student_ckpt = CKPT_DIR / "student.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_models(
        A_tr, B_tr, Y_tr,
        A_val, B_val, Y_val,
        TeacherNet, StudentNet,
        teacher_ckpt, student_ckpt,
        teacher_model_cfg=cfg['teacher_model'],
        student_model_cfg=cfg['student_model'],
        device=device, out_dir=OUT_DIR
    )