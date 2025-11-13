# tuner_optuna.py
# (gamma_feat 튜닝이 추가된 버전)

import os
from pathlib import Path

import optuna
import torch
import yaml

# ❗️[수정] 새 models, trainer, train_utils에서 import
from dataio import load_inputs_and_labels
from models import StudentNet, TeacherNet
from train_utils import make_loaders, metrics_dict
from trainer import train_student_distill


# -------------------------------------------------------------
# Teacher 로드
# -------------------------------------------------------------
def load_teacher(cfg, dim_A, dim_B, device):
    teacher_model_cfg = cfg["teacher_model"].copy()
    teacher_model_cfg.update({
        "dim_A": dim_A,
        "dim_B": dim_B,
        "dim_y": cfg["task"]["dim_y"],
        "num_classes": cfg["task"]["num_classes"]
    })

    # (config.yaml에 지정된 "teacher.pt"를 로드)
    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / cfg["teacher_train"]["ckpt_name"]
    if not ckpt_path.exists():
        print(f"🔥 ERROR: Teacher checkpoint not found at {ckpt_path}")
        print("먼저 'run_train_teacher.py'를 실행하여 teacher.pt를 생성하세요.")
        raise FileNotFoundError(ckpt_path)
        
    teacher = TeacherNet(**teacher_model_cfg).to(device)
    # [수정] torch.load 경고를 피하기 위해 weights_only=True 권장
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    teacher.load_state_dict(state)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Teacher loaded → {ckpt_path}")
    return teacher, teacher_model_cfg


# -------------------------------------------------------------
# Student 평가 (❗️ [핵심 수정] 3-value 반환 처리)
# -------------------------------------------------------------
def evaluate_student(student, val_loader, device, num_classes=2, dim_y=1):
    student.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for a_batch, b_batch, y_batch in val_loader:
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)

            # ❗️ [수정] StudentNet이 (logits, z_a, features) 3개를 반환
            logits, _, _ = student(a_batch) 
            preds = logits.view(-1, dim_y, num_classes).argmax(dim=2)

            y_true_list.append(y_batch)
            y_pred_list.append(preds)

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)

    return metrics_dict(y_true, y_pred)


# -------------------------------------------------------------
# Optuna Objective (❗️ [핵심 수정] gamma_feat 추가)
# -------------------------------------------------------------
def objective(trial):

    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(base / "config.yaml", "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    A, B, Y = load_inputs_and_labels(cfg)
    train_loader, val_loader = make_loaders(A, B, Y, cfg)

    dim_A, dim_B = A.shape[1], B.shape[1]
    num_classes = cfg["task"]["num_classes"]
    dim_y = cfg["task"]["dim_y"]

    teacher, teacher_model_cfg = load_teacher(cfg, dim_A, dim_B, device)

    # --- 1. STUDENT STRUCTURE ---
    z_dim = teacher_model_cfg["z_dim_A"]
    enc_hidden = tuple(teacher_model_cfg["encA_hidden"])

    clf_hidden_str = trial.suggest_categorical("clf_hidden", [
        "(64,)", "(128,)", "(256,)",
        "(64, 32)", "(128, 64)", "(256, 128)",
        "(128, 64, 32)"
    ])
    clf_hidden = eval(clf_hidden_str)

    p_drop = trial.suggest_float("p_drop", 0.05, 0.40)
    use_layernorm = trial.suggest_categorical("use_layernorm", [False, True])

    student_model_cfg = {
        "dim_A": dim_A, "dim_y": dim_y, "num_classes": num_classes,
        "z_dim": z_dim,
        "enc_hidden": enc_hidden,
        "clf_hidden": clf_hidden,
        "p_drop": p_drop,
        "use_layernorm": use_layernorm,
    }

    # --- 2. KD PARAM TUNING (❗️ gamma_feat 추가) ---
    alpha = trial.suggest_float("alpha", 0.1, 0.9) # Hard/Soft 비율
    beta_z = trial.suggest_float("beta_z", 0.1, 2.0) # Z_A 매칭 강도
    temperature = trial.suggest_float("temperature", 1.5, 7.0) # Softness
    
    # ❗️ [추가] 특징 매칭 강도 (가장 중요한 새 파라미터)
    gamma_feat = trial.suggest_loguniform("gamma_feat", 0.1, 10.0) 

    kd_cfg = {
        "alpha": alpha,
        "temperature": temperature,
        "beta_z": beta_z,
        "gamma_feat": gamma_feat # 👈 [추가]
    }

    # --- 3. TRAIN STUDENT ---
    ckpt_dir = str(base / "tune_ckpt_student")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_cfg = {
        **cfg["common_train"],
        **cfg["student_train"], 
        "device": device,
        "ckpt_dir": ckpt_dir,
        "ckpt_name": f"student_trial_{trial.number}.pt",
    }
    try:
        student = train_student_distill(
            A, B, Y,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            model_cfg=student_model_cfg,
            train_cfg=train_cfg,
            kd_cfg=kd_cfg,
            StudentNet=StudentNet, # models.py에서 import한 새 StudentNet
        )

        # --- 4. EVAL ---
        metrics = evaluate_student(student, val_loader, device, num_classes, dim_y)
        score = metrics.get("Accuracy", 0) # Accuracy 기준으로 튜닝

        return score

    except Exception as e:
        print("🔥 TRIAL FAILED! Exact Error:")
        import traceback
        traceback.print_exc()
        return -1e9

# -------------------------------------------------------------
# RUN OPTUNA
# -------------------------------------------------------------
def run_optuna(n_trials=150): # 150회 이상 넉넉하게
    storage_name = "sqlite:///optuna_feature_kd.db"
    study_name = "student_feature_distill_v1"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"Starting Optuna Student (Feature KD) study: {study_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n🔥 Best Params (Feature KD)")
    print(study.best_params)
    print("\n🔥 Best Score (Accuracy)", study.best_value)

    return study


if __name__ == "__main__":
    # (run_train_teacher.py를 먼저 실행해서 teacher.pt를 생성했는지 확인!)
    run_optuna(150)