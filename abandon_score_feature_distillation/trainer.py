# trainer.py
# (train_teacher에 Accuracy 계산 로직 추가)

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ❗️ models에서 새 모델들을 import 합니다.
from models import StudentNet, TeacherNet

# ❗️ [추가] metrics_dict를 import
from train_utils import metrics_dict

# -----------------------------------------------------------------
# 1. train_teacher (❗️ [수정] Accuracy 계산)
# -----------------------------------------------------------------

def train_teacher(A, B, Y, train_loader, val_loader, 
                  model_cfg, train_cfg, TeacherNet, loss_weights={}):
    
    device = train_cfg['device']
    num_classes = model_cfg['num_classes']
    dim_y = model_cfg['dim_y']

    teacher = TeacherNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), 
                            lr=train_cfg['lr'], 
                            weight_decay=train_cfg['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    
    # ❗️ [수정] Loss가 아닌 Accuracy 기준으로 Early Stopping
    best_val_acc = 0.0 
    best_state = None
    patience_counter = 0

    print(f"[Teacher Train] Starting... (Monitoring Val Accuracy)")

    for epoch in range(1, train_cfg['epochs'] + 1):
        teacher.train()
        epoch_loss = 0.0
        
        for a_batch, b_batch, y_batch in train_loader:
            a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
            
            # ❗️ 반환값이 3개 (logits, z_a, features)
            y_pred_logits, _, _ = teacher(a_batch, b_batch)
            
            loss = criterion(y_pred_logits.view(-1, num_classes), y_batch.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * a_batch.size(0)

        # --- Validation Loop (❗️ [수정] Accuracy 계산) ---
        teacher.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], [] # ❗️ [추가]
        
        with torch.no_grad():
            for a_batch, b_batch, y_batch in val_loader:
                a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
                
                # ❗️ 반환값이 3개
                y_pred_logits, _, _ = teacher(a_batch, b_batch)
                loss = criterion(y_pred_logits.view(-1, num_classes), y_batch.view(-1))
                val_loss += loss.item() * a_batch.size(0)
                
                # ❗️ [추가] Accuracy 계산을 위해 예측값과 정답 수집
                preds = y_pred_logits.view(-1, dim_y, num_classes).argmax(dim=2)
                y_true_list.append(y_batch)
                y_pred_list.append(preds)
        
        val_loss /= len(val_loader.dataset)
        
        # ❗️ [추가] Metrics 계산
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        va_metrics = metrics_dict(y_true, y_pred)
        current_val_acc = va_metrics['Accuracy'] # ❗️ Accuracy 사용
        
        if (epoch % 20 == 0) or (epoch == 1):
             # ❗️ [수정] 로그에 Val Acc 추가
             print(f"Epoch {epoch:03d}/{train_cfg['epochs']} | Train Loss: {epoch_loss/len(train_loader.dataset):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {current_val_acc:.4f}")

        # --- Early Stopping (❗️ [수정] Accuracy 기준) ---
        if current_val_acc > best_val_acc: # ❗️ Loss(<)가 아닌 Acc(>) 기준
            best_val_acc = current_val_acc
            best_state = copy.deepcopy(teacher.state_dict())
            patience_counter = 0
            
            # (체크포인트 저장)
            ckpt_dir = Path(train_cfg['ckpt_dir'])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / train_cfg['ckpt_name']
            torch.save(best_state, ckpt_path)

        else:
            patience_counter += 1
            if patience_counter >= train_cfg['early_stop_patience']:
                print(f"Epoch {epoch}: Early stopping triggered. Best Val Acc: {best_val_acc:.4f}")
                break

    print(f"[Teacher Train] Finished. Best Val Acc: {best_val_acc:.4f}")
    
    # Load best model state
    teacher.load_state_dict(best_state)
    return teacher

# -----------------------------------------------------------------
# 2. train_student_distill 함수 (수정 없음)
# -----------------------------------------------------------------
# (이 함수는 KD Loss를 최소화하는 것이 목적이므로 Val Loss 기준으로 둠)

def train_student_distill(A, B, Y, teacher, train_loader, val_loader, 
                          model_cfg, train_cfg, kd_cfg, StudentNet):
    
    device = train_cfg['device']
    num_classes = model_cfg['num_classes']
    dim_y = model_cfg['dim_y']

    # --- 1. KD 하이퍼파라미터 및 손실 함수 정의 ---
    alpha = kd_cfg.get("alpha", 1.0)
    T = kd_cfg.get("temperature", 1.0)
    beta_z = kd_cfg.get("beta_z", 0.0)
    gamma_feat = kd_cfg.get("gamma_feat", 0.0) # 👈 [추가] 특징 손실 가중치

    loss_fn_CE = nn.CrossEntropyLoss()
    loss_fn_KD = nn.KLDivLoss(log_target=True, reduction='batchmean')
    loss_fn_Z = nn.MSELoss()
    loss_fn_Feat = nn.MSELoss() # 👈 [추가] 특징 비교용 손실 (MSE)
    
    student = StudentNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(student.parameters(), 
                            lr=train_cfg['lr'], 
                            weight_decay=train_cfg['weight_decay'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"[Student Train] Starting... (alpha={alpha:.3f}, T={T:.3f}, beta_z={beta_z:.3f}, gamma_feat={gamma_feat:.3f})")

    for epoch in range(1, train_cfg['epochs'] + 1):
        student.train()
        epoch_loss = 0.0
        
        for a_batch, b_batch, y_batch in train_loader:
            a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
            
            with torch.no_grad():
                logits_T, z_a_T, features_T = teacher(a_batch, b_batch)

            logits_S, z_a_S, features_S = student(a_batch)

            # --- 3. 4가지 손실 계산 ---
            loss_CE = loss_fn_CE(logits_S.view(-1, num_classes), y_batch.view(-1))
            loss_KD = loss_fn_KD(
                F.log_softmax(logits_S / T, dim=-1),
                F.log_softmax(logits_T / T, dim=-1)
            ) * (T * T)
            loss_Z = loss_fn_Z(z_a_S, z_a_T)

            loss_Feat = 0.0
            if gamma_feat > 0:
                s_encA_feats = features_S['encA']
                t_encA_feats = features_T['encA']
                
                feat_losses = []
                for s_f, t_f in zip(s_encA_feats, t_encA_feats):
                    feat_losses.append(loss_fn_Feat(s_f, t_f.detach()))
                
                if feat_losses:
                    loss_Feat = torch.stack(feat_losses).mean()

            # --- 4. Total Loss (4개 합산) ---
            total_loss = (alpha * loss_CE) + \
                         ((1 - alpha) * loss_KD) + \
                         (beta_z * loss_Z) + \
                         (gamma_feat * loss_Feat) 

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            epoch_loss += total_loss.item() * a_batch.size(0)

        # --- Validation Loop (Val Loss 기준) ---
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a_batch, b_batch, y_batch in val_loader:
                a_batch, b_batch, y_batch = a_batch.to(device), b_batch.to(device), y_batch.to(device)
                
                logits_T, z_a_T, features_T = teacher(a_batch, b_batch)
                logits_S, z_a_S, features_S = student(a_batch)
                
                l_ce = loss_fn_CE(logits_S.view(-1, num_classes), y_batch.view(-1))
                l_kd = loss_fn_KD(F.log_softmax(logits_S / T, dim=-1), F.log_softmax(logits_T / T, dim=-1)) * (T * T)
                l_z = loss_fn_Z(z_a_S, z_a_T)
                
                l_feat = 0.0
                if gamma_feat > 0:
                    s_encA_feats = features_S['encA']
                    t_encA_feats = features_T['encA']
                    feat_losses = [loss_fn_Feat(s_f, t_f) for s_f, t_f in zip(s_encA_feats, t_encA_feats)]
                    if feat_losses:
                        l_feat = torch.stack(feat_losses).mean()

                l_total = (alpha * l_ce) + ((1 - alpha) * l_kd) + (beta_z * l_z) + (gamma_feat * l_feat)
                val_loss += l_total.item() * a_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        if (epoch % 20 == 0) or (epoch == 1):
             print(f"Epoch {epoch:03d}/{train_cfg['epochs']} | Train Loss: {epoch_loss/len(train_loader.dataset):.4f} | Val Loss: {val_loss:.4f}")

        # --- Early Stopping (Val Loss 기준) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(student.state_dict())
            patience_counter = 0
            
            ckpt_dir = Path(train_cfg['ckpt_dir'])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / train_cfg['ckpt_name']
            torch.save(best_state, ckpt_path)

        else:
            patience_counter += 1
            if patience_counter >= train_cfg['early_stop_patience']:
                print(f"Epoch {epoch}: Early stopping triggered. Best Val Loss: {best_val_loss:.4f}")
                break

    print(f"[Student Train] Finished. Best Val Loss: {best_val_loss:.4f}")
    
    student.load_state_dict(best_state)
    return student