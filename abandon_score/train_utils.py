import json
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split


def resolve_device(device_cfg="auto"):
    """CUDA 사용 가능 여부 자동 감지"""
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg

def make_loaders(A, B, Y, cfg):
    """Numpy 배열을 받아 Train/Validation DataLoader 생성"""
    
    # 1. 텐서로 변환 (Y는 Long 타입)
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)
    
    dataset = TensorDataset(A_t, B_t, Y_t)
    
    # 2. Train / Validation 분할
    n_total = len(dataset)
    n_val = int(n_total * cfg['common_train']['val_split'])
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(cfg['common_train']['seed'])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    # 3. 데이터로더 생성
    bs = cfg['common_train']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    print(f"[Utils] DataLoaders created. Train: {len(train_ds)} samples, Val: {len(val_ds)} samples.")
    return train_loader, val_loader

def metrics_dict(y_true, y_pred_classes):
    """
    분류(Classification)용 평가지표 계산
    y_true: (N, 1) 또는 (N,)
    y_pred_classes: (N, 1) 또는 (N,) (이미 argmax된 클래스)
    """
    y_true_np = y_true.cpu().numpy().squeeze()
    y_pred_np = y_pred_classes.cpu().numpy().squeeze()

    # F1 (Macro): 클래스 불균형이 있더라도 각 클래스를 동일하게 중요시함
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    # Accuracy: 전체 정확도
    accuracy = accuracy_score(y_true_np, y_pred_np)
    
    return {
        "Accuracy": accuracy,
        "F1_Macro": f1_macro
    }

def save_history(history, ckpt_dir, name):
    """훈련 히스토리(Loss)를 JSON과 PNG로 저장"""
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # JSON 저장
        json_path = os.path.join(ckpt_dir, f"{name}_history.json")
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        # Loss Curve PNG 저장
        plt.figure(figsize=(10, 5))
        plt.plot(history['train'], label='Train Loss')
        plt.plot(history['val'], label='Validation Loss')
        plt.title(f'{name} Model Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, f"{name}_loss.png"))
        plt.close()
        
    except Exception as e:
        print(f"[WARN] Failed to save history plot: {e}")