# ============================================================
# trainer_experience.py — Unified (Classification ↔ Regression Auto)
# ============================================================

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # ⬅️ KD에 필요
from train_utils import (  # ⬅️ metrics_dict가 R2/MSE를 반환해야 함
    make_loaders,
    metrics_dict,
    save_history,
)


# ... (유틸 함수들은 동일) ...
def cosine_align_loss(zA, zB, eps=1e-8):
    cos = nn.functional.cosine_similarity(zA, zB, dim=1, eps=eps)
    return (1.0 - cos).mean()

def norm_balance_loss(zA, zB, p=2):
    nA = zA.norm(p=p, dim=1)
    nB = zB.norm(p=p, dim=1)
    return (nA - nB).abs().mean()

def var_balance_loss(zA, zB):
    vA = zA.var(dim=0, unbiased=False)
    vB = zB.var(dim=0, unbiased=False)
    return (vA - vB).abs().mean()

def z_l2_reg(zA, zB):
    if zA is None or zB is None: return 0.0 # 안전장치
    return 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())

# -------------------- TEACHER (회귀 수정) --------------------
def train_teacher(A, B, Y, dim_y, model_cfg, train_cfg, TeacherNet, loss_weights=None):
    """
    [회귀 수정]
    loss = MSE(y_pred, Y_hard)
         + (보조 손실)
    """
    device = train_cfg['device']
    # ❗️ [수정] Y(레이블)는 MSE를 위해 'float' 타입
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    if 'use_layernorm' in model_cfg:
        use_ln = model_cfg['use_layernorm']
    else:
        use_ln = False
    
    # ❗️ [수정] num_classes 제거 (회귀)
    # num_classes = model_cfg['num_classes'] 

    dim_A, dim_B = A.shape[1], B.shape[1]
    model = TeacherNet(dim_A, dim_B, dim_y,
                       encA_hidden=model_cfg['encA_hidden'],
                       encB_hidden=model_cfg['encB_hidden'],
                       clf_hidden=model_cfg['clf_hidden'],
                       z_dim_A=model_cfg['z_dim_A'],
                       z_dim_B=model_cfg['z_dim_B'],
                       p_drop=model_cfg['p_drop'],
                       use_layernorm=use_ln
                       # ❗️ [수정] num_classes 인자 제거
                       ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    
    # ❗️ [수정] CrossEntropyLoss -> MSELoss
    criterion = nn.MSELoss()
    
    tr_loader, va_loader = make_loaders((A,B,Y), batch_size=train_cfg['batch_size'], val_split=train_cfg['val_split'])

    w_align = (loss_weights or {}).get('align', 0.0)
    w_norm = (loss_weights or {}).get('norm_balance', 0.0)
    w_var = (loss_weights or {}).get('var_balance', 0.0)
    w_zl2 = (loss_weights or {}).get('z_l2', 0.0)

    best_val = float('inf'); best_state=None; patience=0
    history = {'train': [], 'val': []}

    for epoch in range(1, train_cfg['epochs'] + 1):
        # ---- train ----
        model.train(); tr_loss = 0.0
        for a, b, y in tr_loader:
            opt.zero_grad()
            y_pred, zA, zB = model(a, b) # y_pred.shape: (Batch, 4)
            y_hard = y                 # y_hard.shape: (Batch, 4)

            # ❗️ [수정] Reshape 필요 없음 (회귀)
            loss_main = criterion(y_pred, y_hard)
            
            loss = loss_main
            if w_align: loss += w_align * cosine_align_loss(zA, zB)
            if w_norm:  loss += w_norm  * norm_balance_loss(zA, zB)
            if w_var:   loss += w_var   * var_balance_loss(zA, zB)
            if w_zl2:   loss += w_zl2   * z_l2_reg(zA, zB)
            
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(a)
        tr_loss /= len(tr_loader.dataset)

        # ---- valid ----
        model.eval(); va_loss=0.0; y_true=[]; y_pred_all=[]
        with torch.no_grad():
            for a, b, y in va_loader:
                yp, zA, zB = model(a, b) # yp.shape: (Batch, 4)
                
                # ❗️ [수정] Loss 계산 (회귀)
                l = criterion(yp, y)
                
                va_loss += l.item() * len(a)
                
                # ❗️ [수정] R2/MSE 계산을 위해 (Batch, 4) 그대로 저장
                y_true.append(y)
                y_pred_all.append(yp)

        va_loss /= len(va_loader.dataset)
        y_true = torch.cat(y_true)
        y_pred_all = torch.cat(y_pred_all)
        va_metrics = metrics_dict(y_true, y_pred_all) # ⬅️ R2/MSE 반환 가정
        
        # ❗️ [수정] R2로 복귀
        acc_key = 'R2'

        print(f"[Teacher][{epoch:03d}] train={tr_loss:.4f} val={va_loss:.4f} "
              f"{acc_key}={va_metrics.get(acc_key, 0.0):.4f}") # ⬅️ R2로 복귀
        history['train'].append(tr_loss); history['val'].append(va_loss)

        if va_loss < best_val - 1e-7:
            best_val = va_loss; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= train_cfg['early_stop_patience']: break

    if best_state: model.load_state_dict(best_state)
    os.makedirs(train_cfg['ckpt_dir'], exist_ok=True)
    
    # ❗️ [수정] 체크포인트 이름 (config.yaml에서 읽어오기)
    ckpt_name = train_cfg.get("ckpt_name", "teacher.pt")
    torch.save(model.state_dict(), os.path.join(train_cfg['ckpt_dir'], ckpt_name))
    save_history(history, train_cfg['ckpt_dir'], ckpt_name.replace('.pt', ''))
    
    return model, history


# -------------------- STUDENT (회귀 + KD 수정) --------------------
def train_student_distill(
    A, B, Y,
    teacher,
    dim_y,
    model_cfg,
    train_cfg,
    StudentNet,
    kd_cfg=None,
    init_from_teacher=True
):
    """
    [회귀 + KD 수정]
    loss = alpha * MSE(y_stu, y_soft)
           + (1 - alpha) * MSE(y_stu, y_hard)
           + beta_z * MSE(z_stu, zA_teacher)
    
    [VALID LOSS]
    val_loss = MSE(y_stu, y_hard)
    """
    device = train_cfg['device']
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    # ❗️ [수정] Y(레이블)는 'float' 타입
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    teacher.eval()
    [p.requires_grad_(False) for p in teacher.parameters()]

    use_ln = model_cfg.get('use_layernorm', False)
    # ❗️ [수정] num_classes 제거
    
    student = StudentNet(
        A.shape[1],
        dim_y,
        enc_hidden=model_cfg['enc_hidden'],
        clf_hidden=model_cfg['clf_hidden'],
        z_dim=model_cfg['z_dim'],
        p_drop=model_cfg['p_drop'],
        use_layernorm=use_ln
        # ❗️ [수정] num_classes 인자 제거
    ).to(device)

    if init_from_teacher:
        try:
            student.init_from_teacher_encoderA(teacher)
            print("[INIT] student.encoder <- teacher.encoder_A")
        except Exception as e:
            print(f"[WARN] init_from_teacher skipped: {e}")

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # ❗️ [수정] CrossEntropyLoss -> MSELoss
    criterion = nn.MSELoss()

    tr_loader, va_loader = make_loaders(
        (A, B, Y),
        batch_size=train_cfg['batch_size'],
        val_split=train_cfg['val_split']
    )

    alpha = train_cfg.get('alpha', 0.7)
    beta_z = (kd_cfg or {}).get('beta_z', 0.0)
    # ❗️ [수정] Temperature(T)는 회귀에서 사용 안 함

    best_val = float('inf')
    best_state = None
    patience = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, train_cfg['epochs'] + 1):
        student.train()
        tr_loss = 0.0

        for a, b, y in tr_loader:
            opt.zero_grad()

            with torch.no_grad():
                y_soft, zA_t, _ = teacher(a, b) # y_soft.shape: (B, 4)

            y_stu, z_stu = student(a) # y_stu.shape: (B, 4)
            y_hard = y                # y_hard.shape: (B, 4)

            # ----------------- Train Loss (회귀 KD) -----------------
            
            # ❗️ [수정] KLD/CE -> MSE
            loss_kd = criterion(y_stu, y_soft)
            loss_task = criterion(y_stu, y_hard)
            
            # 3. Final Loss
            loss = alpha * loss_kd + (1 - alpha) * loss_task
            
            if beta_z:
                loss += beta_z * criterion(z_stu, zA_t) # (z 로스도 MSE)

            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(a)

        tr_loss /= len(tr_loader.dataset)

        # ----------------- Validation (Y_hard 기준) -----------------
        student.eval()
        va_loss = 0.0
        y_true, y_pred_all = [], []

        with torch.no_grad():
            for a, b, y in va_loader:
                y_stu, z_stu = student(a) # y_stu.shape: (B, 4)
                
                # ❗️ [수정] Valid Loss는 오직 Y_hard(실제 정답) 기준 (회귀)
                l = criterion(y_stu, y)
                
                va_loss += l.item() * len(a)
                
                # ❗️ [수정] R2/MSE 계산을 위해 (Batch, 4) 그대로 저장
                y_true.append(y)
                y_pred_all.append(y_stu)

        va_loss /= len(va_loader.dataset)
        y_true = torch.cat(y_true)
        y_pred_all = torch.cat(y_pred_all)
        va_metrics = metrics_dict(y_true, y_pred_all)
        
        # ❗️ [수정] R2로 복귀
        acc_key = 'R2'

        print(
            f"[Student][{epoch:03d}] "
            f"train={tr_loss:.4f} val={va_loss:.4f} "
            f"{acc_key}={va_metrics.get(acc_key, 0.0):.4f}"
        )

        history['train'].append(tr_loss)
        history['val'].append(va_loss)

        # ---------- Early Stopping ----------
        if va_loss < best_val - 1e-7:
            best_val = va_loss
            best_state = copy.deepcopy(student.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg['early_stop_patience']:
                break

    if best_state:
        student.load_state_dict(best_state)

    os.makedirs(train_cfg['ckpt_dir'], exist_ok=True)
    
    # ❗️ [수정] 체크포인트 이름 (config.yaml에서 읽어오기)
    ckpt_name = train_cfg.get("ckpt_name", "student.pt")
    torch.save(
        student.state_dict(),
        os.path.join(train_cfg['ckpt_dir'], ckpt_name)
    )
    save_history(history, train_cfg['ckpt_dir'], ckpt_name.replace('.pt', ''))

    return student, history