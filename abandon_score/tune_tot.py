import os
from pathlib import Path

import optuna
import torch
import yaml
from dataio import load_inputs_and_labels
from models import StudentNet, TeacherNet
from train_utils import make_loaders, metrics_dict
from trainer import train_student_distill


# -------------------------------------------------------------
# Teacher 로드 (수정 없음)
# -------------------------------------------------------------
def load_teacher(cfg, dim_A, dim_B, device):

    teacher_model_cfg = cfg["teacher_model"].copy()
    teacher_model_cfg.update({
        "dim_A": dim_A,
        "dim_B": dim_B,
        "dim_y": cfg["task"]["dim_y"],
        "num_classes": cfg["task"]["num_classes"]
    })

    # (중요!) config.yaml에 정의된 "teacher.pt" (즉, 0.7921짜리 모델)를 로드
    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / cfg["teacher_train"]["ckpt_name"]

    teacher = TeacherNet(**teacher_model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(state)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Teacher (0.7921) loaded → {ckpt_path}")
    return teacher, teacher_model_cfg


# -------------------------------------------------------------
# Student 평가 (수정 없음)
# -------------------------------------------------------------
def evaluate_student(student, val_loader, device, num_classes=2, dim_y=1):
    student.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for a_batch, b_batch, y_batch in val_loader:
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = student(a_batch) # Student는 A만 입력받음
            preds = logits.view(-1, dim_y, num_classes).argmax(dim=2)

            y_true_list.append(y_batch)
            y_pred_list.append(preds)

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)

    return metrics_dict(y_true, y_pred)


# -------------------------------------------------------------
# Optuna Objective (🔥 튜닝 범위 확장)
# -------------------------------------------------------------
def objective(trial):

    # ---------- config load ----------
    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(base / "config.yaml", "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- data load ----------
    A, B, Y = load_inputs_and_labels(cfg)
    train_loader, val_loader = make_loaders(A, B, Y, cfg)

    dim_A = A.shape[1]
    dim_B = B.shape[1]
    num_classes = cfg["task"]["num_classes"]
    dim_y = cfg["task"]["dim_y"]

    # ---------- Teacher load ----------
    # (config.yaml 기반으로 0.7921짜리 Teacher 로드)
    teacher, teacher_model_cfg = load_teacher(cfg, dim_A, dim_B, device)

    # -----------------------------------------------------------------------
    # STUDENT STRUCTURE (Encoder는 Teacher와 완전히 동일하게 고정)
    # -----------------------------------------------------------------------
    
    # (중요!) Teacher의 구조를 그대로 승계 (요청사항 1)
    z_dim = teacher_model_cfg["z_dim_A"]
    enc_hidden = tuple(teacher_model_cfg["encA_hidden"])

    # -----------------------------------------------------------------------
    # 🔥 [수정] Student 고유 파라미터 튜닝 범위 확장 (요청사항 2)
    # -----------------------------------------------------------------------
    
    # [수정] Classifier 구조 (z_dim=32 입력을 고려하여 더 다양하게)
    clf_hidden_str = trial.suggest_categorical("clf_hidden", [
        "(64,)",            # z_dim(32) -> 64 -> N
        "(128,)",           # z_dim(32) -> 128 -> N
        "(256,)",           # z_dim(32) -> 256 -> N
        "(64, 32)",
        "(128, 64)",
        "(256, 128)",
        "(128, 64, 32)"
    ])
    clf_hidden = eval(clf_hidden_str) # 문자열을 튜플로 변환

    # [수정] Dropout 범위 확장
    p_drop = trial.suggest_float("p_drop", 0.05, 0.40) # (기존: 0.05 ~ 0.25)
    
    use_layernorm = trial.suggest_categorical("use_layernorm", [False, True]) # (기존과 동일)

    student_model_cfg = {
        "dim_A": dim_A,
        "dim_y": dim_y,
        "num_classes": num_classes,
        "z_dim": z_dim,           # ⬅️ Teacher와 동기화
        "enc_hidden": enc_hidden, # ⬅️ Teacher와 동기화
        "clf_hidden": clf_hidden, # ⬅️ Student 고유 튜닝
        "p_drop": p_drop,         # ⬅️ Student 고유 튜닝
        "use_layernorm": use_layernorm,
    }

    # -----------------------------------------------------------------------
    # 🔥 [수정] KD PARAM TUNING 범위 확장 (요청사항 2)
    # -----------------------------------------------------------------------
    
    # [수정] alpha (Hard/Soft Label 비율)
    # Teacher가 79%이므로, Hard Label(진짜 정답)을 좀 더 신뢰할 여지를 줌
    alpha = trial.suggest_float("alpha", 0.50, 0.95) # (기존: 0.70 ~ 0.95)

    # [수정] temperature 범위 확장
    temperature = trial.suggest_float("temperature", 1.5, 7.0) # (기존: 1.5 ~ 5.0)

    # [수정] beta_z (Latent Space 매칭)
    # z_dim=32라는 핵심 정보를 잘 배우도록 가중치 범위를 대폭 확장
    beta_z = trial.suggest_float("beta_z", 0.1, 2.0) # (기존: 0.00 ~ 0.20)

    kd_cfg = {
        "alpha": alpha,
        "temperature": temperature,
        "beta_z": beta_z
    }

    # -----------------------------------------------------------------------
    # TRAIN STUDENT
    # -----------------------------------------------------------------------
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
            StudentNet=StudentNet,
        )

        # ---------- eval ----------
        metrics = evaluate_student(student, val_loader, device, num_classes, dim_y)
        
        # [수정] Teacher 튜닝 때와 동일하게 "Accuracy"를 기준으로 평가
        score = metrics.get("Accuracy", 0) 

        return score

    except Exception as e:
        print("🔥 TRIAL FAILED! Exact Error:")
        import traceback
        traceback.print_exc()    # 전체 스택 출력
        
        return -1e9      # 실패한 Trial은 매우 낮은 점수 반환

# -------------------------------------------------------------
# RUN OPTUNA (🔥 [수정] Trial 횟수, Study 이름)
# -------------------------------------------------------------
def run_optuna(n_trials=150): # 👈 [수정] 탐색 공간이 넓어졌으니 150회 이상 추천
    
    # [수정] DB에 저장하고, 튜닝 이름 변경
    storage_name = "sqlite:///optuna_student.db"
    study_name = "student_tune_with_79_teacher"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"Starting Optuna Student study: {study_name} [Storage: {storage_name}]")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n🔥 Best Params (Student with 79% Teacher)")
    print(study.best_params)
    print("\n🔥 Best Score (Accuracy)", study.best_value)

    return study


if __name__ == "__main__":
    run_optuna(150) # 👈 [수정] 150회 실행