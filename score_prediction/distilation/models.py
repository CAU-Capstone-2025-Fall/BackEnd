import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧩 기본 MLP 블록 (BatchNorm 사용 버전)
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128),
                 p_drop=0.1, use_layernorm=False):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers += [nn.LeakyReLU(0.1, inplace=True),
                       nn.Dropout(p_drop)]
            last = h

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 🧠 TeacherNet
# ============================================================
class TeacherNet(nn.Module):
    def __init__(self, dim_A, dim_B, dim_y,
                 encA_hidden=(256, 128), encB_hidden=(256, 128),
                 clf_hidden=(256, 128),
                 z_dim_A=64, z_dim_B=64,
                 p_drop=0.1, use_layernorm=False,
                 num_classes=1):  # ✅ 기본값 1 (회귀용)
        super().__init__()

        self.dim_y = dim_y
        self.num_classes = num_classes

        # --- Encoder 정의
        self.encoder_A = MLP(dim_A, z_dim_A, encA_hidden, p_drop, use_layernorm)
        self.encoder_B = MLP(dim_B, z_dim_B, encB_hidden, p_drop, use_layernorm)

        # --- 출력 차원 계산 로직 수정
        if num_classes > 1:
            out_dim = dim_y * num_classes   # 분류 문제
        else:
            out_dim = dim_y                 # 회귀 문제

        self.classifier = MLP(
            z_dim_A + z_dim_B,
            out_dim,
            clf_hidden,
            p_drop,
            use_layernorm
        )

    def forward(self, A, B):
        zA = self.encoder_A(A)
        zB = self.encoder_B(B)
        z = torch.cat([zA, zB], dim=1)
        y = self.classifier(z)
        return y, zA, zB


# ============================================================
# 🎓 StudentNet
# ============================================================
class StudentNet(nn.Module):
    def __init__(self, dim_A, dim_y,
                 enc_hidden=(256, 128),
                 clf_hidden=(256, 128),
                 z_dim=64,
                 p_drop=0.1,
                 use_layernorm=False,
                 num_classes=1):  # ✅ 기본값 1 (회귀용)
        super().__init__()

        self.dim_y = dim_y
        self.num_classes = num_classes

        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)

        # --- 출력 차원 계산 로직 수정
        if num_classes > 1:
            out_dim = dim_y * num_classes
        else:
            out_dim = dim_y

        self.classifier = MLP(
            z_dim,
            out_dim,
            clf_hidden,
            p_drop,
            use_layernorm
        )

    def forward(self, A):
        z = self.encoder(A)
        y = self.classifier(z)
        return y, z

    def init_from_teacher_encoderA(self, teacher):
        import copy as _copy
        self.encoder.load_state_dict(_copy.deepcopy(teacher.encoder_A.state_dict()))
