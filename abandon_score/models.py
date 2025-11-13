import copy

import torch
import torch.nn as nn


# ============================================================
# 🧩 1. 기본 MLP 블록 (BatchNorm1d / LayerNorm 선택)
# ============================================================
class MLP(nn.Module):
    """
    MLP 블록 (BatchNorm1d 또는 LayerNorm 지원)
    순서: Linear -> Norm -> Activation -> Dropout
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(128,), 
                 p_drop=0.1, use_layernorm=False):
        super().__init__()
        layers = []
        last_dim = in_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            else:
                # 1D 데이터(테이블)에는 BatchNorm1d가 더 안정적
                layers.append(nn.BatchNorm1d(h_dim))
                
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p_drop))
            last_dim = h_dim
            
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ============================================================
# 🧠 2. Teacher Network (A+B 입력)
# ============================================================
class TeacherNet(nn.Module):
    def __init__(self, dim_A, dim_B, dim_y=1, num_classes=2,
                 encA_hidden=(128,), encB_hidden=(128,),
                 clf_hidden=(128, 64),
                 z_dim_A=32, z_dim_B=32,
                 p_drop=0.1, use_layernorm=False):
        super().__init__()
        
        # A, B 인코더
        self.encoder_A = MLP(dim_A, z_dim_A, encA_hidden, p_drop, use_layernorm)
        self.encoder_B = MLP(dim_B, z_dim_B, encB_hidden, p_drop, use_layernorm)
        
        # 분류기 (Classifier)
        # 최종 출력: (dim_y * num_classes) (예: 1 * 2 = 2)
        self.classifier = MLP(
            z_dim_A + z_dim_B, 
            dim_y * num_classes, 
            clf_hidden, 
            p_drop, 
            use_layernorm
        )

    def forward(self, A, B):
        zA = self.encoder_A(A)
        zB = self.encoder_B(B)
        z_fused = torch.cat([zA, zB], dim=1)
        
        # (Batch, 2) 크기의 원시(raw) Logits 반환
        logits = self.classifier(z_fused) 
        return logits, zA, zB

# ============================================================
# 🎓 3. Student Network (A 입력)
# ============================================================
class StudentNet(nn.Module):
    def __init__(self, dim_A, dim_y=1, num_classes=2,
                 enc_hidden=(128,), clf_hidden=(128, 64),
                 z_dim=32, p_drop=0.1, use_layernorm=False):
        super().__init__()
        
        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)
        self.classifier = MLP(
            z_dim, 
            dim_y * num_classes, 
            clf_hidden, 
            p_drop, 
            use_layernorm
        )

    def forward(self, A):
        z = self.encoder(A)
        logits = self.classifier(z) # (Batch, 2) Logits 반환
        return logits, z

    def init_from_teacher_encoderA(self, teacher):
        """Teacher의 Encoder_A 가중치를 복사"""
        try:
            self.encoder.load_state_dict(copy.deepcopy(teacher.encoder_A.state_dict()))
            print("[INFO] Student.encoder weights initialized from Teacher.encoder_A.")
        except Exception as e:
            print(f"[WARN] Failed to initialize Student from Teacher. Mismatch? {e}")