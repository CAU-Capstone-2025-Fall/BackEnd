# ============================================================
# evaluation_utils.py
#  - TeacherNet latent space 품질 평가 모듈 (+ Student alignment plot)
# ============================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import linear_kernel


# ---------------------------
# 📊 Metric Functions
# ---------------------------
def cka(X, Y):
    X -= X.mean(0); Y -= Y.mean(0)
    K = linear_kernel(X); L = linear_kernel(Y)
    hsic = (K * L).sum()
    return hsic / (np.linalg.norm(K) * np.linalg.norm(L))

def latent_variance_ratio(zA, zB):
    varA, varB = zA.var(axis=0).mean(), zB.var(axis=0).mean()
    return (varA + varB) / (np.var(np.concatenate([zA, zB], axis=1)))

def latent_norm_ratio(zA, zB):
    normA, normB = np.mean(np.linalg.norm(zA, axis=1)), np.mean(np.linalg.norm(zB, axis=1))
    return normA / (normB + 1e-8)

def cross_corr_mean(zA, zB):
    corr = np.corrcoef(np.concatenate([zA.T, zB.T], axis=0))
    nA, nB = zA.shape[1], zB.shape[1]
    cross_block = corr[:nA, nA:]
    return np.nanmean(np.abs(cross_block))

def tsa_score(zA, Z):
    """Teacher–Student Alignment proxy: zA vs concat(zA,zB) similarity"""
    return cka(zA, Z)


# ---------------------------
# 🧠 Evaluation Routine
# ---------------------------
def evaluate_teacher_latent(teacher, A, B, save_dir="./analysis_results"):
    import torch
    teacher.eval()
    with torch.no_grad():
        A_t = torch.tensor(A, dtype=torch.float32, device=next(teacher.parameters()).device)
        B_t = torch.tensor(B, dtype=torch.float32, device=next(teacher.parameters()).device)
        _, zA, zB = teacher(A_t, B_t)

    zA, zB = zA.cpu().numpy(), zB.cpu().numpy()
    Z = np.concatenate([zA, zB], axis=1)

    results = {
        "LVR": latent_variance_ratio(zA, zB),
        "LNR": latent_norm_ratio(zA, zB),
        "CCM": cross_corr_mean(zA, zB),
        "CKA_zA_zB": cka(zA, zB),
        "TSA": tsa_score(zA, Z),
    }

    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "teacher_latent_metrics.csv")
    df.to_csv(csv_path, index=False)

    # --- correlation heatmap ---
    corr = np.corrcoef(np.concatenate([zA.T, zB.T], axis=0))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("zA–zB Cross-Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "teacher_latent_corr.png"))
    plt.close()

    print(f"[EVAL] Teacher latent quality metrics saved → {csv_path}")
    print(df)


# ---------------------------
# 🎯 Teacher–Student Alignment Plot
# ---------------------------
def plot_teacher_student_alignment(teacher, student, A, B=None, save_dir="./analysis_results"):
    """Compare teacher zA vs student z alignment."""
    import torch
    teacher.eval(); student.eval()
    device = next(teacher.parameters()).device

    with torch.no_grad():
        A_t = torch.tensor(A, dtype=torch.float32, device=device)
        if B is not None:
            B_t = torch.tensor(B, dtype=torch.float32, device=device)
            _, zA, _ = teacher(A_t, B_t)
        else:
            # 만약 B가 주어지지 않았다면 A를 임시로 대입 (차원 맞을 때만)
            _, zA, _ = teacher(A_t, A_t)
        # Student는 A만 받는 구조 가정
        out = student(A_t)
        zS = out[1] if isinstance(out, tuple) and len(out) > 1 else out

    zA, zS = zA.cpu().numpy(), zS.cpu().numpy()

    # 차원 맞추기
    min_dim = min(zA.shape[1], zS.shape[1])
    zA, zS = zA[:, :min_dim], zS[:, :min_dim]

    # 상관행렬 및 평균 CKA 계산
    align_corr = np.corrcoef(np.concatenate([zA.T, zS.T], axis=0))
    cka_score = cka(zA, zS)

    os.makedirs(save_dir, exist_ok=True)
    sns.heatmap(align_corr, cmap="coolwarm", center=0)
    plt.title(f"Teacher–Student Alignment (CKA={cka_score:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "teacher_student_alignment.png"))
    plt.close()

    print(f"[PLOT] Saved teacher–student alignment → {save_dir}/teacher_student_alignment.png")
    print(f"[METRIC] CKA(teacher zA, student z) = {cka_score:.4f}")
    return cka_score
