# tuner_optuna.py
# (❗️ [핵심 수정] 재현성 보장 + lr/wd 튜닝 + 빡센 탐색)

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
        "dim_A": dim_A, "dim_B": dim_B,
        "dim_y": cfg["task"]["dim_y"], "num_classes": cfg["task"]["num_classes"]
    })

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / cfg["teacher_train"]["ckpt_name"]
    if not ckpt_path.exists():
        print(f"🔥 ERROR: Teacher checkpoint not found at {ckpt_path}")
        print("먼저 'run_distill.py'를 실행하여 85.15%짜리 teacher.pt를 생성하세요.")
        raise FileNotFoundError(ckpt_path)
        
    teacher = TeacherNet(**teacher_model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    teacher.load_state_dict(state)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ❗️ [수정] 요청하신 대로 85.15% 로드 메시지 제거
    # print(f"[INFO] Teacher (85.15%) loaded → {ckpt_path}") 
    return teacher, teacher_model_cfg


# -------------------------------------------------------------
# Student 평가 (3-value 반환 처리)
# -------------------------------------------------------------
def evaluate_student(student, val_loader, device, num_classes=2, dim_y=1):
    student.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for a_batch, b_batch, y_batch in val_loader:
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _, _ = student(a_batch) # 3-value
            preds = logits.view(-1, dim_y, num_classes).argmax(dim=2)
            y_true_list.append(y_batch)
            y_pred_list.append(preds)

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    return metrics_dict(y_true, y_pred)


# -------------------------------------------------------------
# Optuna Objective (❗️ [핵심 수정] 빡세게 튜닝 + 시드 고정)
# -------------------------------------------------------------
def objective(trial):

    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(base / "config.yaml", "r", encoding="utf-8"))

    # ❗️ [필수] 재현성을 위해 모든 트라이얼의 시드를 고정!
    torch.manual_seed(cfg['common_train']['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    A, B, Y = load_inputs_and_labels(cfg)
    train_loader, val_loader = make_loaders(A, B, Y, cfg)

    dim_A, dim_B = A.shape[1], B.shape[1]
    num_classes = cfg["task"]["num_classes"]
    dim_y = cfg["task"]["dim_y"]

    teacher, teacher_model_cfg = load_teacher(cfg, dim_A, dim_B, device)

    # --- 1. STUDENT STRUCTURE ---
    
    # ❗️ [고정] z_dim, enc_hidden은 Feature KD를 위해 Teacher와 동일해야 함
    z_dim = teacher_model_cfg["z_dim_A"]
    enc_hidden = tuple(teacher_model_cfg["encA_hidden"])

    # ❗️ [빡센 튜닝] Student의 Classifier만 튜닝
    clf_hidden_str = trial.suggest_categorical("clf_hidden", [
        "(64,)", "(128,)", "(256,)", "(512,)",
        "(128, 64)", "(256, 128)", "(512, 256)",
        "(256, 128, 64)", "(512, 256, 128)"
    ])
    clf_hidden = eval(clf_hidden_str)

    # ❗️ [빡센 튜닝] p_drop 범위 확장
    p_drop = trial.suggest_float("p_drop", 0.05, 0.5)
    use_layernorm = trial.suggest_categorical("use_layernorm", [False, True])

    student_model_cfg = {
        "dim_A": dim_A, "dim_y": dim_y, "num_classes": num_classes,
        "z_dim": z_dim,
        "enc_hidden": enc_hidden,
        "clf_hidden": clf_hidden,
        "p_drop": p_drop,
        "use_layernorm": use_layernorm,
    }

    # --- 2. ❗️ [빡센 튜닝] OPTIMIZER & KD PARAMS ---
    
    # ❗️ [추가] Student의 lr/wd도 튜닝
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # ❗️ [빡센 튜닝] KD 파라미터 범위 확장
    alpha = trial.suggest_float("alpha", 0.05, 0.95) 
    beta_z = trial.suggest_float("beta_z", 0.1, 5.0, log=True) 
    temperature = trial.suggest_float("temperature", 1.5, 10.0)
    gamma_feat = trial.suggest_float("gamma_feat", 0.1, 20.0, log=True) 

    kd_cfg = {
        "alpha": alpha,
        "temperature": temperature,
        "beta_z": beta_z,
        "gamma_feat": gamma_feat
    }

    # --- 3. TRAIN STUDENT ---
    ckpt_dir = str(base / "tune_ckpt_student")
    # os.makedirs(ckpt_dir, exist_ok=True) # (파일 상단에서 이미 생성함)
    
    train_cfg = {
        **cfg["common_train"],
        # ❗️ [수정] 튜닝된 lr/wd 사용
        "lr": lr,
        "weight_decay": weight_decay,
        
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
            StudentNet=StudentNet,
        )

        # --- 4. EVAL ---
        metrics = evaluate_student(student, val_loader, device, num_classes, dim_y)
        score = metrics.get("Accuracy", 0) 

        # ❗️ [추가] 요청하신 대로 매 트라이얼 종료 시 Accuracy 출력
        print(f"--- Trial {trial.number} FINISHED --- Score (Accuracy): {score:.6f}")

        return score

    except Exception as e:
        print(f"🔥 TRIAL FAILED! Exact Error:")
        import traceback
        traceback.print_exc()
        return -1e9

# -------------------------------------------------------------
# RUN OPTUNA
# -------------------------------------------------------------
def run_optuna(n_trials=200): # ❗️ 150 -> 200회 "빡세게"
    storage_name = "sqlite:///optuna_feature_kd.db"
    
    # ❗️[수정] lr/wd 튜닝을 포함하는 새 Study 이름
    study_name = "student_tune_full_vs_85_teacher_v5" # v2 -> v3
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"Starting Optuna Student (Full Tune vs 85% Teacher, Seed Fixed) study: {study_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n🔥 Best Params ({study_name})")
    print(study.best_params)
    print("\n🔥 Best Score (Accuracy)", study.best_value)

    return study


if __name__ == "__main__":
    # ❗️(85.15% 'teacher.pt'가 checkpoints 폴더에 있는지 확인!)
    run_optuna(200) # ❗️ 200회