import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import metrics_dict, save_history


# -------------------- 1. TEACHER (분류) --------------------
def train_teacher(A, B, Y, train_loader, val_loader, model_cfg, train_cfg, TeacherNet, loss_weights=None):
    
    device = train_cfg['device']
    num_classes = model_cfg['num_classes']
    dim_y = model_cfg['dim_y']
    
    model = TeacherNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    w_zl2 = (loss_weights or {}).get('z_l2', 0.0)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train': [], 'val': []}

    print(f"[INFO] Starting Teacher Training ({train_cfg['epochs']} epochs)...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        tr_loss = 0.0
        for a_batch, b_batch, y_batch in train_loader:
            a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
            
            opt.zero_grad()
            y_pred_logits, zA, zB = model(a_batch, b_batch)
            
            # (B, 2) -> (B*1, 2) | (B, 1) -> (B*1)
            loss_main = criterion(y_pred_logits.view(-1, num_classes), y_batch.view(-1))
            loss = loss_main
            
            if w_zl2 > 0:
                loss += w_zl2 * 0.5 * (zA.pow(2).mean() + zB.pow(2).mean())
            
            loss.backward()
            opt.step()
            tr_loss += loss.item() * a_batch.size(0)
        
        tr_loss /= len(train_loader.dataset)
        history['train'].append(tr_loss)

        # ---- Validation ----
        model.eval()
        va_loss = 0.0
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for a_batch, b_batch, y_batch in val_loader:
                a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
                
                yp_logits, _, _ = model(a_batch, b_batch)
                l = criterion(yp_logits.view(-1, num_classes), y_batch.view(-1))
                va_loss += l.item() * a_batch.size(0)
                
                # Accuracy/F1 계산용
                yp_classes = yp_logits.view(-1, dim_y, num_classes).argmax(dim=2)
                y_true_all.append(y_batch)
                y_pred_all.append(yp_classes)

        va_loss /= len(val_loader.dataset)
        history['val'].append(va_loss)
        
        va_metrics = metrics_dict(torch.cat(y_true_all), torch.cat(y_pred_all))

        print(
            f"[Teacher][{epoch:03d}/{train_cfg['epochs']}] "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {va_loss:.4f} | "
            f"Val Acc: {va_metrics['Accuracy']:.4f} | "
            f"Val F1: {va_metrics['F1_Macro']:.4f}"
        )

        # Early Stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['early_stop_patience']:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    
    ckpt_dir = train_cfg['ckpt_dir']
    ckpt_name = train_cfg['ckpt_name']
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
    save_history(history, ckpt_dir, ckpt_name.replace('.pt', ''))
    
    print(f"✅ Teacher training complete. Model saved to {ckpt_dir}/{ckpt_name}")
    return model

# -------------------- 2. STUDENT (분류 + KD) --------------------
def train_student_distill(A, B, Y, teacher, train_loader, val_loader, model_cfg, train_cfg, kd_cfg, StudentNet):
    
    device = train_cfg['device']
    num_classes = model_cfg['num_classes']
    dim_y = model_cfg['dim_y']
    
    teacher.eval()
    [p.requires_grad_(False) for p in teacher.parameters()]
    
    student = StudentNet(**model_cfg).to(device)
    student.init_from_teacher_encoderA(teacher) # 가중치 초기화
    
    opt = torch.optim.AdamW(student.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    
    criterion_task = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    criterion_feat = nn.MSELoss() # (beta_z용)
    
    alpha = kd_cfg['alpha']
    beta_z = kd_cfg['beta_z']
    T = kd_cfg['temperature']

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train': [], 'val': []}

    print(f"[INFO] Starting Student Training ({train_cfg['epochs']} epochs)...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        student.train()
        tr_loss = 0.0
        for a_batch, b_batch, y_batch in train_loader:
            a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
            
            with torch.no_grad():
                y_soft, zA_t, _ = teacher(a_batch, b_batch)
            
            y_stu, z_stu = student(a_batch)
            y_hard = y_batch
            
            # --- 1. Task Loss (Hard Label) ---
            loss_task = criterion_task(
                y_stu.view(-1, num_classes), 
                y_hard.view(-1)
            )
            
            # --- 2. KD Loss (Soft Label) ---
            y_soft_probs = F.softmax(y_soft.view(-1, num_classes) / T, dim=-1)
            y_stu_log_probs = F.log_softmax(y_stu.view(-1, num_classes) / T, dim=-1)
            loss_kd = criterion_kd(y_stu_log_probs, y_soft_probs) * (T * T)
            
            # --- 3. Final Loss ---
            loss = (alpha * loss_kd) + ((1 - alpha) * loss_task)
            
            if beta_z > 0:
                loss += beta_z * criterion_feat(z_stu, zA_t)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * a_batch.size(0)
        
        tr_loss /= len(train_loader.dataset)
        history['train'].append(tr_loss)

        # ---- Validation (Y_hard 기준) ----
        student.eval()
        va_loss = 0.0
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for a_batch, b_batch, y_batch in val_loader:
                a_batch, y_batch = a_batch.to(device), y_batch.to(device)
                
                yp_logits, _ = student(a_batch)
                
                # Valid Loss는 오직 Y_hard(실제 정답) 기준
                l = criterion_task(yp_logits.view(-1, num_classes), y_batch.view(-1))
                va_loss += l.item() * a_batch.size(0)
                
                yp_classes = yp_logits.view(-1, dim_y, num_classes).argmax(dim=2)
                y_true_all.append(y_batch)
                y_pred_all.append(yp_classes)

        va_loss /= len(val_loader.dataset)
        history['val'].append(va_loss)
        
        va_metrics = metrics_dict(torch.cat(y_true_all), torch.cat(y_pred_all))

        print(
            f"[Student][{epoch:03d}/{train_cfg['epochs']}] "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {va_loss:.4f} | "
            f"Val Acc: {va_metrics['Accuracy']:.4f} | "
            f"Val F1: {va_metrics['F1_Macro']:.4f}"
        )

        # Early Stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['early_stop_patience']:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    if best_state:
        student.load_state_dict(best_state)

    ckpt_dir = train_cfg['ckpt_dir']
    ckpt_name = train_cfg['ckpt_name']
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(ckpt_dir, ckpt_name))
    save_history(history, ckpt_dir, ckpt_name.replace('.pt', ''))
    
    print(f"✅ Student training complete. Model saved to {ckpt_dir}/{ckpt_name}")
    return student