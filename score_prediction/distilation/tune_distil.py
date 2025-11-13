# ============================================================
# 🎯 Teacher Model Hyperparameter Optimization (with dataio.py)
# ============================================================

import json
import os

import numpy as np
import optuna
import torch
import yaml
from dataio import load_config, load_inputs_and_labels, resolve_device
from models import TeacherNet
from trainer import train_teacher

# ------------------------------------------------------------
# 0️⃣ Config & Data Load
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

cfg = load_config(CONFIG_PATH)
device = resolve_device(cfg.get("device", "auto"))

A, B, Y = load_inputs_and_labels(cfg)
dim_y = Y.shape[1]
dim_A, dim_B = A.shape[1], B.shape[1]

print(f"[INFO] Loaded data: A={A.shape}, B={B.shape}, Y={Y.shape}")


# ------------------------------------------------------------
# 1️⃣ Optuna Objective
# ------------------------------------------------------------
def objective(trial):
    # ----- Hyperparameters to search -----
    z_dim_A = trial.suggest_int("z_dim_A", 16, 128, step=16)
    z_dim_B = trial.suggest_int("z_dim_B", 16, 128, step=16)

    enc_scale = trial.suggest_categorical("enc_scale", [0.5, 1.0, 2.0])
    clf_scale = trial.suggest_categorical("clf_scale", [0.5, 1.0, 2.0])

    base_encA = np.array(cfg["teacher_model"]["encA_hidden"])
    base_encB = np.array(cfg["teacher_model"]["encB_hidden"])
    base_clf = np.array(cfg["teacher_model"]["clf_hidden"])

    encA_hidden = tuple((base_encA * enc_scale).astype(int))
    encB_hidden = tuple((base_encB * enc_scale).astype(int))
    clf_hidden = tuple((base_clf * clf_scale).astype(int))

    p_drop = trial.suggest_float("p_drop", 0.05, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)

    # ----- Model & Train Configs -----
    model_cfg = dict(
        encA_hidden=encA_hidden,
        encB_hidden=encB_hidden,
        clf_hidden=clf_hidden,
        z_dim_A=z_dim_A,
        z_dim_B=z_dim_B,
        p_drop=p_drop,
        use_layernorm=False,
        num_classes=cfg["teacher_model"]["num_classes"],
    )

    train_cfg = cfg["teacher_train"] | {
        "device": device,
        "lr": lr,
        "weight_decay": weight_decay,
        "ckpt_dir": os.path.join(BASE_DIR, "tune_teacher_ckpt"),
    }

    # ----- Train & Evaluate -----
    teacher, hist = train_teacher(
        A, B, Y, dim_y,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        TeacherNet=TeacherNet,
    )

    best_val = min(hist["val"])
    return best_val


# ------------------------------------------------------------
# 2️⃣ Run Optuna Study
# ------------------------------------------------------------
STUDY_NAME = "teacher_tuning"
STORAGE = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_teacher.db')}"
N_TRIALS = 20

study = optuna.create_study(
    direction="minimize",
    study_name=STUDY_NAME,
    storage=STORAGE,
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS)

# ------------------------------------------------------------
# 3️⃣ Save Best Result
# ------------------------------------------------------------
print("\n===== 🧠 Best Teacher Trial =====")
best = study.best_trial
for k, v in best.params.items():
    print(f"  {k}: {v}")

os.makedirs(os.path.join(BASE_DIR, "tune_logs"), exist_ok=True)
with open(os.path.join(BASE_DIR, "tune_logs", "best_teacher_trial.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_params": best.params,
            "best_value": best.value,
            "device": device,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[DONE] 최적 Teacher 파라미터 저장 완료 → tune_logs/best_teacher_trial.json")
