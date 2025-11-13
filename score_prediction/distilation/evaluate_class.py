# ============================================================
# evaluate_with_weaklabels.py (분류용 + feature_cols 버그 수정)
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
from models import MLP, StudentNet, TeacherNet
from torch.utils.data import TensorDataset, random_split

# ⚠️ 'train_utils'와 'models'가 임포트 가능해야 함
from train_utils import metrics_dict

# ❗️ [수정] feature_cols를 모든 함수가 접근 가능한 전역 상수로 정의
FEATURE_COLS = [
   "experience"
]

# ============================================================
# 1️⃣ Validation 복원 + weak label 매칭
# ============================================================
def load_holdout_data(A, B, label_path, val_split=0.2, seed=42):
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)

    df_label = pd.read_excel(str(label_path))
    
    # ❗️ [수정] 전역 상수 FEATURE_COLS 사용
    assert len(A_t) == len(df_label), "A/B와 weak_labels.xlsx의 샘플 수가 다릅니다."
    Y_t = torch.tensor(df_label[FEATURE_COLS].values, dtype=torch.long)
    ds = TensorDataset(A_t, B_t, Y_t)

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_ds, va_ds = random_split(ds, [n_train, n_val], generator=gen)

    A_tr = torch.stack([s[0] for s in tr_ds])
    B_tr = torch.stack([s[1] for s in tr_ds])
    Y_tr = torch.stack([s[2] for s in tr_ds])

    A_val = torch.stack([s[0] for s in va_ds])
    B_val = torch.stack([s[1] for s in va_ds])
    Y_val = torch.stack([s[2] for s in va_ds])

    print(f"[LOAD] Train set: {len(A_tr)}, Validation set: {len(A_val)}")
    return A_tr, B_tr, Y_tr, A_val, B_val, Y_val


# ============================================================
# 2️⃣ Baseline (A only) (분류용 수정)
# ============================================================
class BaselineNet(nn.Module):
    def __init__(self, dim_A, dim_y, enc_hidden, clf_hidden, z_dim, 
                 p_drop=0.1, use_layernorm=False, num_classes=3):
        super().__init__()
        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)
        self.classifier = MLP(
            z_dim, 
            dim_y * num_classes, 
            clf_hidden, 
            p_drop, 
            use_layernorm
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.classifier(z)
        return y, z


# ============================================================
# 3️⃣ 세 모델 평가 (분류용 수정)
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
    A_tr, B_tr = A_tr.to(device), B_tr.to(device)
    Y_tr = Y_tr.to(device, dtype=torch.long)
    A_val, B_val = A_val.to(device), B_val.to(device)
    Y_val = Y_val.to(device, dtype=torch.long)
    
    num_classes = teacher_model_cfg["num_classes"]
    dim_y = teacher_model_cfg["dim_y"]

    # --- 1. Teacher
    with torch.no_grad():
        teacher = TeacherNet(
            dim_A=teacher_model_cfg["dim_A"],
            dim_B=teacher_model_cfg["dim_B"],
            dim_y=dim_y,
            encA_hidden=teacher_model_cfg["encA_hidden"],
            encB_hidden=teacher_model_cfg["encB_hidden"],
            clf_hidden=teacher_model_cfg["clf_hidden"],
            z_dim_A=teacher_model_cfg["z_dim_A"],
            z_dim_B=teacher_model_cfg["z_dim_B"],
            p_drop=teacher_model_cfg["p_drop"],
            use_layernorm=teacher_model_cfg["use_layernorm"],
            num_classes=num_classes
        ).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        Yp_teacher_logits, _, _ = teacher(A_val, B_val)
        Yp_teacher = Yp_teacher_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_teacher = metrics_dict(Y_val, Yp_teacher)

    # --- 2. Student
    with torch.no_grad():
        student = StudentNet(
            dim_A=student_model_cfg["dim_A"],
            dim_y=dim_y,
            enc_hidden=student_model_cfg["enc_hidden"],
            clf_hidden=student_model_cfg["clf_hidden"],
            z_dim=student_model_cfg["z_dim"],
            p_drop=student_model_cfg["p_drop"],
            use_layernorm=student_model_cfg["use_layernorm"],
            num_classes=num_classes
        ).to(device)
        student.load_state_dict(torch.load(student_ckpt, map_location=device, weights_only=True))
        student.eval()
        Yp_student_logits, _ = student(A_val)
        Yp_student = Yp_student_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_student = metrics_dict(Y_val, Yp_student)

    # --- 3. Baseline
    baseline = BaselineNet(
        dim_A=student_model_cfg["dim_A"],
        dim_y=dim_y,
        enc_hidden=student_model_cfg["enc_hidden"],
        clf_hidden=student_model_cfg["clf_hidden"],
        z_dim=student_model_cfg["z_dim"],
        p_drop=student_model_cfg["p_drop"],
        use_layernorm=student_model_cfg["use_layernorm"],
        num_classes=num_classes
    ).to(device)

    opt = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 50 

    for ep in range(epochs):
        baseline.train()
        opt.zero_grad()
        y_pred_logits, _ = baseline(A_tr)
        y_pred_reshaped = y_pred_logits.view(-1, num_classes)
        y_hard_reshaped = Y_tr.view(-1)
        loss = criterion(y_pred_reshaped, y_hard_reshaped)
        loss.backward()
        opt.step()

    with torch.no_grad():
        baseline.eval()
        Yp_base_logits, _ = baseline(A_val)
        Yp_base = Yp_base_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_base = metrics_dict(Y_val, Yp_base)

    # --- 4. 결과 요약
    df = pd.DataFrame([
        {"Model": "Teacher (A+B)", **metrics_teacher},
        {"Model": "Student (Distilled)", **metrics_student},
        {"Model": "Baseline (A only)", **metrics_base},
    ])
    df.to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)
    print("\n===== 📊 Evaluation Summary =====")
    print(df)

    # --- 5. 시각화 (분류용)
    try:
        metrics = [k for k in metrics_base.keys() if 'acc' in k.lower() or 'f1' in k.lower()]
        if not metrics: metrics = list(metrics_base.keys())
    except Exception:
        metrics = list(metrics_base.keys())
        
    palette = ["#ff7f0e", "#1f77b4", "#2ca02c"]

    plt.figure(figsize=(max(5, len(metrics)*2.5), 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        sns.barplot(x="Model", y=metric, hue="Model", data=df, palette=palette, legend=False, order=df['Model'])
        plt.title(metric)
        plt.xticks(rotation=25, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Model Performance on Hold-out Validation Set")
    plt.savefig(os.path.join(out_dir, "eval_barplots.png"), dpi=200)
    plt.close()

    # --- 6. 예측값 저장 (분류용)
    # ❗️ [수정] 전역 상수 FEATURE_COLS 사용
    df_pred_true = pd.DataFrame(Y_val.cpu().numpy(), columns=FEATURE_COLS)
    df_pred_teacher = pd.DataFrame(Yp_teacher.cpu().numpy(), columns=[f"pred_T_{c}" for c in FEATURE_COLS])
    df_pred_student = pd.DataFrame(Yp_student.cpu().numpy(), columns=[f"pred_S_{c}" for c in FEATURE_COLS])
    df_pred_baseline = pd.DataFrame(Yp_base.cpu().numpy(), columns=[f"pred_B_{c}" for c in FEATURE_COLS])
    
    df_pred_all = pd.concat([df_pred_true, df_pred_teacher, df_pred_student, df_pred_baseline], axis=1)
    df_pred_all.to_csv(os.path.join(out_dir, "val_predictions_classification.csv"), index=False)
    
    print(f"[SAVE] All results saved → {out_dir}")

    return df


# ============================================================
# 4️⃣ 실행 예시 (분류용 수정)
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

    # --- 데이터 로드 ---
    try:
        A = pd.read_csv(DATA_DIR / "A_noex.csv").to_numpy()
        B = pd.read_csv(DATA_DIR / "B_noex.csv").to_numpy()
    except FileNotFoundError:
        A = np.load(DATA_DIR / "A.npy")
        B = np.load(DATA_DIR / "B.npy")

    # --- Y 레이블 파일 탐색 ---
    Y_candidates = cfg['filenames']['Y_candidates']
    label_path = None
    search_dirs = [DATA_DIR, DATA_DIR / "label"]
    
    for sdir in search_dirs:
        for cand in Y_candidates:
            for ext in ['.xlsx', '.csv', '.npy']:
                p = sdir / (cand + ext)
                if p.exists():
                    label_path = p
                    print(f"[LOAD] Found Y label file: {p}")
                    break
            if label_path: break
        if label_path: break

    if label_path is None:
        raise FileNotFoundError(f"Could not find any Y label file from {Y_candidates} in {search_dirs}")


    # --- Train/Validation 데이터 로드 (long 타입 Y 포함) ---
    A_tr, B_tr, Y_tr, A_val, B_val, Y_val = load_holdout_data(
        A, B, label_path, 
        val_split=cfg['teacher_train']['val_split'], 
        seed=cfg['seed']
    )

    # --- config 딕셔너리에 dim 정보 주입 ---
    cfg['teacher_model'].update({
        "dim_A": A.shape[1],
        "dim_B": B.shape[1],
        "dim_y": Y_val.shape[1], # (dim_y는 6)
    })
    cfg['student_model'].update({
        "dim_A": A.shape[1],
        "dim_y": Y_val.shape[1], # (dim_y는 6)
    })

    # --- config 키 존재 여부 확인 ---
    for key in ['num_classes', 'use_layernorm']:
        if key not in cfg['teacher_model']:
            raise KeyError(f"'{key}' missing from teacher_model in config.yaml")
        if key not in cfg['student_model']:
            cfg['student_model'][key] = cfg['teacher_model'][key] # Teacher 설정 동기화


    teacher_ckpt = CKPT_DIR / "teacher.pt"
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