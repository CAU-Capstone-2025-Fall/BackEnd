import json
import math
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split


# -------------------------------------------------
# 📊 Metric 계산 (회귀 + 분류 공용)
# -------------------------------------------------
def metrics_dict(y_true, y_pred):
    """
    y_true, y_pred: torch.Tensor
    분류(Classification) 또는 회귀(Regression)에 모두 대응
    """
    # Tensor 강제 변환 + CPU로 이동
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    y_true, y_pred = y_true.detach().cpu(), y_pred.detach().cpu()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    metrics = {}

    # --- 회귀 지표 ---
    try:
        y_true_f, y_pred_f = y_true.float(), y_pred.float()
        mse = torch.mean((y_true_f - y_pred_f) ** 2).item()
        mae = torch.mean((y_true_f - y_pred_f).abs()).item()
    except Exception:
        mse, mae = math.nan, math.nan

    metrics['MSE'] = mse
    metrics['MAE'] = mae

    # --- 분류 지표 ---
    # 정수형 라벨일 경우만 계산
    if y_true.dtype in (torch.int32, torch.int64):
        acc = (y_true == y_pred).float().mean().item()
        metrics['Accuracy'] = acc

        try:
            f1_macro = f1_score(y_true.numpy(), y_pred.numpy(), average="macro")
            f1_micro = f1_score(y_true.numpy(), y_pred.numpy(), average="micro")
            metrics['Macro_F1'] = f1_macro
            metrics['Micro_F1'] = f1_micro
        except Exception:
            metrics['Macro_F1'] = math.nan
            metrics['Micro_F1'] = math.nan
    else:
        metrics['Accuracy'] = math.nan
        metrics['Macro_F1'] = math.nan
        metrics['Micro_F1'] = math.nan

    return metrics


# -------------------------------------------------
# 🔄 Train/Val Loader 생성
# -------------------------------------------------
def make_loaders(tensors, batch_size=128, val_split=0.2, seed=42):
    """훈련/검증 데이터로더 생성 (안정적 random split 포함)."""
    ds = TensorDataset(*tensors)
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))  # ✅ 최소 1개 보장
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(ds, [n_train, n_val], generator=gen)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return tr_loader, va_loader


# -------------------------------------------------
# 🧾 학습 기록 저장 + 손실 그래프 저장
# -------------------------------------------------
def save_history(history, path, prefix="teacher"):
    """학습/검증 손실 기록 및 곡선 저장."""
    if history is None or "train" not in history or "val" not in history:
        print(f"[WARN] Invalid history object. Skipping save for {prefix}.")
        return

    os.makedirs(path, exist_ok=True)

    # JSON 저장
    hist_path = os.path.join(path, f"{prefix}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] History JSON → {hist_path}")

    # 손실 그래프
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(history["train"], label="Train", color="orange", linewidth=2)
        plt.plot(history["val"], label="Validation", color="green", linewidth=2)
        plt.title(f"{prefix.capitalize()} Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(path, f"{prefix}_loss.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[SAVE] Loss curve image → {fig_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot history curve: {e}")
