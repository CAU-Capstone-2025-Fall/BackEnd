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
from models import StudentNet, TeacherNet
from train_utils import make_loaders, metrics_dict, resolve_device
from trainer import train_student_distill

# ------------------------------------------------------------
# 0. 기본 설정 및 데이터 로드
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
os.makedirs(os.path.join(BASE_DIR, "tune_ckpt_student"), exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DEVICE = resolve_device(CFG['common_train']['device'])
torch.manual_seed(CFG['common_train']['seed'])

A, B, Y = load_inputs_and_labels(CFG)
dim_A, dim_B = A.shape[1], B.shape[1]
dim_y, num_classes = CFG['task']['dim_y'], CFG['task']['num_classes']

# 훈련/검증 로더 (모든 Trial이 공유)
train_loader, val_loader = make_loaders(A, B, Y, CFG)

print(f"[INFO] Data loaded. Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
print(f"[INFO] Using device: {DEVICE}")

# ------------------------------------------------------------
# 1. 🥇 "최고의 교사" (TeacherNet, Acc 84.1%) 로드
# ------------------------------------------------------------
teacher_ckpt_path = BASE_DIR / CFG['paths']['ckpt_dir'] / CFG['teacher_train']['ckpt_name']
if not teacher_ckpt_path.exists():
    raise FileNotFoundError(f"❌ {teacher_ckpt_path} - 최적화된 교사 모델이 없습니다. run_distill.py를 먼저 실행하세요.")

# (config.yaml에서 Teacher 설정 로드)
teacher_model_cfg = CFG['teacher_model'].copy()
teacher_model_cfg.update({"dim_A": dim_A, "dim_B": dim_B, "dim_y": dim_y, "num_classes": num_classes})

teacher = TeacherNet(**teacher_model_cfg).to(DEVICE)
teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=DEVICE, weights_only=True))
teacher.eval()
[p.requires_grad_(False) for p in teacher.parameters()]

print(f"✅ Best Teacher loaded from {teacher_ckpt_path} (Acc 84.1%)")

# ------------------------------------------------------------
# 2. Optuna Objective 정의 (Student 훈련 + Accuracy 반환)
# ------------------------------------------------------------
def objective(trial):
    """
    StudentNet의 'Validation Accuracy'를 최대화하는 하이퍼파라미터를 찾습니다.
    """
    print(f"=== Trial {trial.number} START ===")
    try:
        student, hist = train_student_distill(...)
        print("hist:", hist)
        print("val history:", hist.get("val"))
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        raise e
    # --- 1. 하이퍼파라미터 제안 ---
    
    # KD 파라미터 (Student 성능을 58% -> 70%+로 올릴 핵심)
    alpha = trial.suggest_float("alpha", 0.7, 0.99, log=True)
    temperature = trial.suggest_float("temperature", 2.0, 7.0)
    
    # 훈련 파라미터
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    p_drop = trial.suggest_float("p_drop", 0.1, 0.4)

    # 아키텍처 파라미터 (A만으로 A+B를 함축해야 하므로 Teacher보다 복잡할 수 있음)
    z_dim = trial.suggest_categorical("z_dim", [32, 64, 128])
    n_layers_enc = trial.suggest_int("n_layers_enc", 1, 2)
    enc_hidden = [128, 64] if n_layers_enc == 2 else [128]
    n_layers_clf = trial.suggest_int("n_layers_clf", 1, 2)
    clf_hidden = [128, 64] if n_layers_clf == 2 else [128]

    # --- 2. 설정 구성 ---
    model_cfg = {
        "dim_A": dim_A, "dim_y": dim_y, "num_classes": num_classes,
        "z_dim": z_dim,
        "enc_hidden": tuple(enc_hidden),
        "clf_hidden": tuple(clf_hidden),
        "p_drop": p_drop,
        "use_layernorm": CFG["student_model"]["use_layernorm"],
    }
    
    train_cfg = CFG['common_train'].copy()
    train_cfg.update(CFG['student_train'])
    train_cfg.update({
        "device": DEVICE,
        "lr": lr,
        "weight_decay": weight_decay,
        "ckpt_dir": os.path.join(BASE_DIR, "tune_ckpt_student"),
        "ckpt_name": f"trial_{trial.number}_student.pt"  # ★ 필수
    })

    
    kd_cfg = {
        "alpha": alpha,
        "temperature": temperature,
        "beta_z": CFG['student_kd']['beta_z'] # (0으로 고정)
    }

    # Teacher.Encoder_A와 구조가 일치할 때만 가중치 초기화
    init_from_teacher = (
        z_dim == CFG["teacher_model"]["z_dim_A"] and
        tuple(enc_hidden) == tuple(CFG["teacher_model"]["encA_hidden"])
    )

    # --- 3. 훈련 실행 (trainer.py 호출) ---
    # (trainer.py가 Val Acc/F1을 계산하고, Val Loss를 기준으로 Early Stopping함)
    
    try:
        # ❗️ [수정] trainer.py를 직접 호출 (내부 루프 대신)
        student, hist = train_student_distill(
            A, B, Y, 
            teacher=teacher,
            train_loader=train_loader, 
            val_loader=val_loader,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            kd_cfg=kd_cfg,
            StudentNet=StudentNet,
            init_from_teacher=init_from_teacher
        )
        
        # ❗️ [수정] trainer.py가 Val_Loss를 기준으로 Early Stopping하고
        # 그 때의 val_loss를 반환한다고 가정 (hist['val']의 마지막 값)
        
        # (만약 trainer.py가 metrics_dict를 반환하지 않으면, 여기서 직접 계산해야 함)
        # (지금은 trainer.py가 val_loss를 반환한다고 가정)
        best_val_loss = min(hist["val"])
        return best_val_loss # ❗️ [수정] "손실(Loss) 최소화"를 목표로 함

    except Exception as e:
        print(f"⚠️ Trial failed: {e}")
        return float('inf') # (손실 최소화이므로, 실패 시 무한대)

# ------------------------------------------------------------
# 3. Optuna 실행
# ------------------------------------------------------------
STUDY_NAME = "student_distill_tune_v2" # ❗️ 스터디 이름
STORAGE = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_student.db')}"
N_TRIALS = 50  # ⬅️ 시도 횟수 (50회 이상 추천)

shutil.rmtree(os.path.join(BASE_DIR, "tune_ckpt_student"), ignore_errors=True)

study = optuna.create_study(
    direction="minimize", # ❗️ [수정] "Validation Loss"를 "최소화"
    study_name=STUDY_NAME,
    storage=STORAGE,
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS)

# ------------------------------------------------------------
# 4. 결과 저장
# ------------------------------------------------------------
print("\n" + "="*30)
print("===== 🎓 Best Student Trial (Min Val Loss) =====")
best = study.best_trial
print(f"  Value (Min Val Loss): {best.value:.6f}")
print("  Params: ")
for k, v in best.params.items():
    print(f"    {k}: {v}")

os.makedirs(os.path.join(BASE_DIR, "tune_logs"), exist_ok=True)
best_params_path = os.path.join(BASE_DIR, "tune_logs", "best_student_params.json")
with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_params": best.params,
            "best_value(MinValLoss)": best.value,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[DONE] 최적 Student 파라미터 저장 완료 → {best_params_path}")
print("이제 이 파라미터를 config.yaml의 student_... 섹션에 반영하고,")
print("[run_distill.py] (Phase 2만) 및 [run_evaluation.py]를 실행하여 최종 성능을 확인하세요.")