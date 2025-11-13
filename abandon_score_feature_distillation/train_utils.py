# train_utils.py

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, random_split


def resolve_device(device_str="auto"):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def make_loaders(A, B, Y, cfg):
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)
    dataset = TensorDataset(A_t, B_t, Y_t)
    
    n_total = len(dataset)
    n_val = int(n_total * cfg['common_train']['val_split'])
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(cfg['common_train']['seed'])
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['common_train']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg['common_train']['batch_size'], 
        shuffle=False
    )
    return train_loader, val_loader

def metrics_dict(y_true, y_pred):
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    
    return {
        "Accuracy": accuracy_score(y_true_np, y_pred_np),
        "F1_Macro": f1_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "Precision": precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "Recall": recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
    }