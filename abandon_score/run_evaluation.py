import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from dataio import load_inputs_and_labels
from models import MLP, StudentNet, TeacherNet
from torch.utils.data import (  # (Split은 필요 없음)
    DataLoader,
    TensorDataset,
    random_split,
)
from train_utils import make_loaders, metrics_dict, resolve_device


# ============================================================
# 1. Baseline (A only) 모델 정의
# ============================================================
class BaselineNet(nn.Module):
    def __init__(self, dim_A, dim_y, num_classes, enc_hidden, clf_hidden, z_dim, 
                 p_drop=0.1, use_layernorm=False):
        super().__init__()
        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)
        self.classifier = MLP(z_dim, dim_y * num_classes, clf_hidden, p_drop, use_layernorm)

    def forward(self, x):
        z = self.encoder(x)
        y = self.classifier(z)
        return y, z

# ============================================================
# 2. 메인 평가 함수
# ============================================================
def evaluate_all_models(cfg, A_tr, B_tr, Y_tr, A_val, B_val, Y_val, device):
    
    ckpt_dir = cfg['paths']['ckpt_dir']
    eval_dir = cfg['paths']['eval_dir']
    os.makedirs(eval_dir, exist_ok=True)
    
    dim_A, dim_B, dim_y, num_classes = A_tr.shape[1], B_tr.shape[1], cfg['task']['dim_y'], cfg['task']['num_classes']
    
    # (모델 Config 준비)
    teacher_model_cfg = cfg['teacher_model'].copy()
    teacher_model_cfg.update({"dim_A": dim_A, "dim_B": dim_B, "dim_y": dim_y, "num_classes": num_classes})
    
    student_model_cfg = cfg['student_model'].copy()
    student_model_cfg.update({"dim_A": dim_A, "dim_y": dim_y, "num_classes": num_classes})

    # (데이터 Device로 이동)
    A_tr, Y_tr = A_tr.to(device), Y_tr.to(device)
    A_val, B_val, Y_val = A_val.to(device), B_val.to(device), Y_val.to(device)

    all_metrics = []

    # --- 1. Teacher (A+B) 평가 ---
    print("[EVAL] Evaluating Teacher (A+B)...")
    with torch.no_grad():
        teacher = TeacherNet(**teacher_model_cfg).to(device)
        teacher_ckpt = os.path.join(ckpt_dir, cfg['teacher_train']['ckpt_name'])
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        
        Yp_teacher_logits, _, _ = teacher(A_val, B_val)
        Yp_teacher_classes = Yp_teacher_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_teacher = metrics_dict(Y_val, Yp_teacher_classes)
        all_metrics.append({"Model": "Teacher (A+B)", **metrics_teacher})

    # --- 2. Student (Distilled) 평가 ---
    print("[EVAL] Evaluating Student (A only, Distilled)...")
    with torch.no_grad():
        student = StudentNet(**student_model_cfg).to(device)
        student_ckpt = os.path.join(ckpt_dir, cfg['student_train']['ckpt_name'])
        student.load_state_dict(torch.load(student_ckpt, map_location=device, weights_only=True))
        student.eval()
        
        Yp_student_logits, _ = student(A_val)
        Yp_student_classes = Yp_student_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_student = metrics_dict(Y_val, Yp_student_classes)
        all_metrics.append({"Model": "Student (Distilled)", **metrics_student})

    # --- 3. Baseline (A only) 훈련 및 평가 ---
    print("[EVAL] Training and Evaluating Baseline (A only)...")
    baseline = BaselineNet(
        dim_A=dim_A, dim_y=dim_y, num_classes=num_classes,
        enc_hidden=student_model_cfg["enc_hidden"],
        clf_hidden=student_model_cfg["clf_hidden"],
        z_dim=student_model_cfg["z_dim"],
        p_drop=student_model_cfg["p_drop"],
        use_layernorm=student_model_cfg["use_layernorm"]
    ).to(device)
    
    # (Baseline은 Student와 동일한 하이퍼파라미터로 "제대로" 훈련)
    opt = torch.optim.AdamW(
        baseline.parameters(), 
        lr=cfg['student_train']['lr'], 
        weight_decay=cfg['student_train']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # (DataLoader로 A_tr, Y_tr 묶기)
    baseline_train_ds = TensorDataset(A_tr, Y_tr)
    baseline_train_loader = DataLoader(baseline_train_ds, batch_size=cfg['common_train']['batch_size'], shuffle=True)
    
    for ep in range(cfg['common_train']['epochs']): # (Student/Teacher와 동일한 Epoch)
        baseline.train()
        for a_batch, y_batch in baseline_train_loader:
            opt.zero_grad()
            y_pred_logits, _ = baseline(a_batch)
            loss = criterion(y_pred_logits.view(-1, num_classes), y_batch.view(-1))
            loss.backward()
            opt.step()
        
        # (간단한 로그)
        if (ep + 1) % 50 == 0:
            print(f"  [Baseline] Epoch {ep+1} complete, Loss: {loss.item():.4f}")

    with torch.no_grad():
        baseline.eval()
        Yp_base_logits, _ = baseline(A_val)
        Yp_base_classes = Yp_base_logits.view(-1, dim_y, num_classes).argmax(dim=2)
        metrics_base = metrics_dict(Y_val, Yp_base_classes)
        all_metrics.append({"Model": "Baseline (A only)", **metrics_base})

    # --- 4. 결과 요약 및 저장 ---
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(eval_dir, "eval_summary.csv"), index=False)
    print("\n" + "="*50)
    print("===== 📊 FINAL EVALUATION SUMMARY =====")
    print(df.round(4))
    print("="*50 + "\n")

    # --- 5. 시각화 ---
    metrics_to_plot = list(metrics_base.keys())
    palette = ["#ff7f0e", "#1f77b4", "#2ca02c"]
    
    plt.figure(figsize=(max(5, len(metrics_to_plot)*4), 5))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), i + 1)
        sns.barplot(x="Model", y=metric, hue="Model", data=df, palette=palette, legend=False, order=df['Model'])
        plt.title(metric)
        plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Final Model Performance Comparison")
    plt.savefig(os.path.join(eval_dir, "eval_barplots.png"), dpi=200)
    plt.close()
    
    print(f"[SAVE] All results saved → {eval_dir}")
    return df

# ============================================================
# 3. 실행 예시
# ============================================================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    
    # --- 1. Load Config ---
    cfg_path = base / "config.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    DEVICE = resolve_device(cfg['common_train']['device'])
    torch.manual_seed(cfg['common_train']['seed'])

    # --- 2. Load Data ---
    A, B, Y = load_inputs_and_labels(cfg)
    
    # --- 3. Get Train/Val Datasets (Numpy/Torch) ---
    # (evaluate_models가 Baseline 훈련을 위해 Train 셋이 필요함)
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)
    dataset = TensorDataset(A_t, B_t, Y_t)
    
    n_total = len(dataset)
    n_val = int(n_total * cfg['common_train']['val_split'])
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(cfg['common_train']['seed'])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    # (A, B, Y를 훈련/검증용 텐서로 분리)
    A_tr = torch.stack([s[0] for s in train_ds])
    B_tr = torch.stack([s[1] for s in train_ds])
    Y_tr = torch.stack([s[2] for s in train_ds])
    
    A_val = torch.stack([s[0] for s in val_ds])
    B_val = torch.stack([s[1] for s in val_ds])
    Y_val = torch.stack([s[2] for s in val_ds])

    # --- 4. Run Evaluation ---
    evaluate_all_models(
        cfg, 
        A_tr, B_tr, Y_tr,
        A_val, B_val, Y_val,
        device=DEVICE
    )