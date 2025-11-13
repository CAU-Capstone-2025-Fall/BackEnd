# run_train_teacher.py
# (Feature Distillation과 호환되는 teacher.pt를 새로 생성)

import os
from pathlib import Path

import torch
import yaml

# ❗️[중요] 새 폴더의 파일들을 import
from dataio import load_inputs_and_labels
from models import TeacherNet
from train_utils import make_loaders, resolve_device
from trainer import train_teacher


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
    
    teacher_train_cfg = {
        **cfg['common_train'], 
        **cfg['teacher_train'], 
        "device": DEVICE, 
        "ckpt_dir": cfg['paths']['ckpt_dir']
    }

    # --- 5. Train Teacher ---
    print("\n" + "="*50)
    print("🚀 Training NEW Teacher for Feature Distillation...")
    print("="*50)
    
    # (trainer.py의 train_teacher 함수를 호출하여 훈련 및 저장)
    teacher = train_teacher(
        A, B, Y, train_loader, val_loader,
        model_cfg=teacher_model_cfg,
        train_cfg=teacher_train_cfg,
        TeacherNet=TeacherNet, # models.py의 새 TeacherNet
    )
    
    print("\n✅ New 'teacher.pt' created successfully.")
    print(f"File saved at: {Path(teacher_train_cfg['ckpt_dir']) / teacher_train_cfg['ckpt_name']}")

if __name__ == "__main__":
    main()