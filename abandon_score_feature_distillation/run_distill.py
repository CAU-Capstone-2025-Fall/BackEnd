# run_distill.py
# (featuredistillation 폴더의 models.py, trainer.py를 사용)

import os
from pathlib import Path

import torch
import yaml

# [수정] 새 models, trainer, train_utils를 import
from dataio import load_inputs_and_labels
from models import StudentNet, TeacherNet
from train_utils import make_loaders, resolve_device
from trainer import train_student_distill, train_teacher


def main():
    base = Path(__file__).resolve().parent
    
    # --- 1. Load Config ---
    cfg_path = base / "config.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    DEVICE = resolve_device(cfg['common_train']['device'])
    torch.manual_seed(cfg['common_train']['seed'])

    # --- 2. Load Data ---
    A, B, Y = load_inputs_and_labels(cfg)
    
    # --- 3. Create DataLoaders ---
    train_loader, val_loader = make_loaders(A, B, Y, cfg)
    
    # --- 4. Prepare Configs ---
    dim_A, dim_B = A.shape[1], B.shape[1]
    dim_y, num_classes = cfg['task']['dim_y'], cfg['task']['num_classes']
    
    teacher_model_cfg = cfg['teacher_model'].copy()
    teacher_model_cfg.update({"dim_A": dim_A, "dim_B": dim_B, "dim_y": dim_y, "num_classes": num_classes})
    
    student_model_cfg = cfg['student_model'].copy()
    student_model_cfg.update({"dim_A": dim_A, "dim_y": dim_y, "num_classes": num_classes})
    
    teacher_train_cfg = {
        **cfg['common_train'], 
        **cfg['teacher_train'], 
        "device": DEVICE, 
        "ckpt_dir": cfg['paths']['ckpt_dir']
    }
    student_train_cfg = {
        **cfg['common_train'], 
        **cfg['student_train'], 
        "device": DEVICE, 
        "ckpt_dir": cfg['paths']['ckpt_dir']
    }

    # --- 5. Train Teacher ---
    print("\n" + "="*50)
    print("🚀 PHASE 1: Training Teacher Network (Feature KD Version)...")
    print("="*50)
    # (trainer.py의 train_teacher (Accuracy 기준) 호출)
    teacher = train_teacher(
        A, B, Y, train_loader, val_loader,
        model_cfg=teacher_model_cfg,
        train_cfg=teacher_train_cfg,
        TeacherNet=TeacherNet, # models.py의 새 TeacherNet
    )
    
    # --- 6. Train Student ---
    print("\n" + "="*50)
    print("🎓 PHASE 2: Training Student Network (Feature KD)...")
    print("="*50)
    # (trainer.py의 train_student_distill (Val Loss 기준) 호출)
    student = train_student_distill(
        A, B, Y, 
        teacher=teacher, # ❗️ 방금 훈련된 Teacher를 그대로 전달
        train_loader=train_loader, 
        val_loader=val_loader,
        model_cfg=student_model_cfg,
        train_cfg=student_train_cfg,
        kd_cfg=cfg['student_kd'],
        StudentNet=StudentNet # models.py의 새 StudentNet
    )

    print("\n✅ All training phases complete.")
    print(f"Teacher saved to: {teacher_train_cfg['ckpt_dir']}/{teacher_train_cfg['ckpt_name']}")
    print(f"Student saved to: {student_train_cfg['ckpt_dir']}/{student_train_cfg['ckpt_name']}")

if __name__ == "__main__":
    main()