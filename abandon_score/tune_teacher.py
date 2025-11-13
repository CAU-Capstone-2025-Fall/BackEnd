import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from dataio import load_inputs_and_labels
from models import TeacherNet
from train_utils import make_loaders, metrics_dict, resolve_device

# ------------------------------------------------------------
# 0. 기본 설정 및 데이터 로드
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEVICE = resolve_device(CFG['common_train']['device'])
torch.manual_seed(CFG['common_train']['seed'])

A, B, Y = load_inputs_and_labels(CFG)
dim_A, dim_B = A.shape[1], B.shape[1]
dim_y, num_classes = CFG['task']['dim_y'], CFG['task']['num_classes']

# ❗️ 훈련/검증 로더를 여기서 한 번만 생성
train_loader, val_loader = make_loaders(A, B, Y, CFG)

print(f"[INFO] Data loaded. Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Target Metric: Accuracy") # 👈 [수정]

# ------------------------------------------------------------
# 2. Optuna Objective 정의 (Teacher 훈련 + Accuracy 반환)
# ------------------------------------------------------------
def objective(trial):
    """
    TeacherNet의 'Validation Accuracy'를 최대화하는 하이퍼파라미터를 찾습니다.
    """
    
    # --- 1. 하이퍼파라미터 제안 ---
    
    # Training
    lr = trial.suggest_loguniform("lr", 1e-4, 3e-3) 
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    p_drop = trial.suggest_float("p_drop", 0.1, 0.4)
    
    # Structure
    z_dim = trial.suggest_categorical("z_dim", [32, 64, 128]) 
    
    enc_hidden_str = trial.suggest_categorical("enc_hidden", [
        "(128,)", 
        "(256,)", 
        "(128, 64)", 
        "(256, 128)",
        "(256, 128, 64)" 
    ])
    enc_hidden = eval(enc_hidden_str)

    clf_hidden_str = trial.suggest_categorical("clf_hidden", [
        "(128,)", 
        "(256,)", 
        "(128, 64)", 
        "(256, 128)",
        "(256, 128, 64)"
    ])
    clf_hidden = eval(clf_hidden_str)

    use_layernorm = trial.suggest_categorical("use_layernorm", [True, False])

    # --- 2. 설정 구성 ---
    model_cfg = {
        "dim_A": dim_A, "dim_B": dim_B, "dim_y": dim_y, "num_classes": num_classes,
        "z_dim_A": z_dim, "z_dim_B": z_dim,
        "encA_hidden": enc_hidden,
        "encB_hidden": enc_hidden,
        "clf_hidden": clf_hidden,
        "p_drop": p_drop,
        "use_layernorm": use_layernorm, 
    }
    
    train_cfg = CFG['common_train'].copy()
    train_cfg.update(CFG['teacher_train']) 
    train_cfg.update({
        "device": DEVICE,
        "lr": lr,
        "weight_decay": weight_decay,
        "ckpt_dir": os.path.join(BASE_DIR, "tune_ckpt_teacher")
    })

    # --- 3. 훈련 실행 ---
    model = TeacherNet(**model_cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_score = 0.0 # 👈 [수정] Accuracy 기준
    best_state = None
    patience_counter = 0

    try:
        for epoch in range(1, train_cfg['epochs'] + 1):
            model.train()
            for a_batch, b_batch, y_batch in train_loader:
                a_batch, b_batch, y_batch = a_batch.to(DEVICE), b_batch.to(DEVICE), y_batch.to(DEVICE)
                opt.zero_grad()
                y_pred_logits, _, _ = model(a_batch, b_batch)
                loss = criterion(y_pred_logits.view(-1, num_classes), y_batch.view(-1))
                loss.backward()
                opt.step()

            # ---- Validation (매 에포크마다 Acc 계산) ----
            model.eval()
            y_true_all, y_pred_all = [], []
            with torch.no_grad():
                for a_batch, b_batch, y_batch in val_loader:
                    a_batch, b_batch, y_batch = a_batch.to(DEVICE), b_batch.to(DEVICE), y_batch.to(DEVICE)
                    yp_logits, _, _ = model(a_batch, b_batch)
                    
                    yp_classes = yp_logits.view(-1, dim_y, num_classes).argmax(dim=2)
                    y_true_all.append(y_batch)
                    y_pred_all.append(yp_classes)
            
            va_metrics = metrics_dict(torch.cat(y_true_all), torch.cat(y_pred_all))
            
            # 👈 [수정] Accuracy를 목표 점수로 사용
            current_val_score = va_metrics.get('Accuracy', 0.0) 
            
            # 👈 [수정] 최고 "Accuracy"를 기준으로 모델 저장
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_cfg['early_stop_patience']:
                    break 

        # 👈 [수정] 최고 "Accuracy"를 Optuna에 반환
        return best_val_score

    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0 

# ------------------------------------------------------------
# 3. Optuna 실행
# ------------------------------------------------------------
# 👈 [수정] 스터디 이름 변경 (Accuracy 명시)
STUDY_NAME = "teacher_full_structure_tune_acc_v1" 
STORAGE = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_teacher.db')}"
N_TRIALS = 100  

shutil.rmtree(os.path.join(BASE_DIR, "tune_ckpt_teacher"), ignore_errors=True)
os.makedirs(os.path.join(BASE_DIR, "tune_ckpt_teacher"), exist_ok=True)

study = optuna.create_study(
    direction="maximize", # "Accuracy"를 "최대화"
    study_name=STUDY_NAME,
    storage=STORAGE,
    load_if_exists=True,
)
print(f"Starting Optuna study: {STUDY_NAME} [Storage: {STORAGE}]")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True) 

# ------------------------------------------------------------
# 4. 결과 저장
# ------------------------------------------------------------
print("\n" + "="*30)
print("===== 🧠 Best Teacher Trial (Accuracy) =====") # 👈 [수정]
best = study.best_trial
print(f"  Value (Max Val Accuracy): {best.value:.6f}") # 👈 [수정]
print("  Params: ")
for k, v in best.params.items():
    print(f"    {k}: {v}")

os.makedirs(os.path.join(BASE_DIR, "tune_logs"), exist_ok=True)
# 👈 [수정] 파일 이름 변경 (acc 명시)
best_params_path = os.path.join(BASE_DIR, "tune_logs", "best_teacher_params_acc.json") 
with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_params": best.params,
            "best_value(Accuracy)": best.value, # 👈 [수정]
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\n[DONE] 최적 Teacher 파라미터 저장 완료 → {best_params_path}")
print("이제 이 파라미터를 config.yaml에 반영하고 [run_distill.py]를 실행하여 'teacher.pt'를 생성하세요.")
print("그 다음, [run_optuna_student.py]를 실행하여 학생을 튜닝하세요.")

if __name__ == "__main__":
    pass