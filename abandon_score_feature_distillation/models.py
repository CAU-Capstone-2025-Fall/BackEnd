# models.py
# (Feature Distillation을 위해 중간 특징을 반환하는 모델)

import torch
import torch.nn as nn


# -----------------------------------------------------------------
# 1. [핵심 수정] MLP 모듈: 중간 특징을 반환하도록 변경
# -----------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dims, p_drop=0.1, use_layernorm=False):
        super().__init__()
        # ❗️ nn.Sequential 대신 nn.ModuleList 사용
        self.layers = nn.ModuleList()
        current_dim = dim_in
        
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            if use_layernorm:
                self.layers.append(nn.LayerNorm(h_dim))
            # ❗️ 활성화 함수도 모듈로 추가
            self.layers.append(nn.ReLU()) 
            self.layers.append(nn.Dropout(p_drop))
            current_dim = h_dim
        
        # 마지막 출력 레이어
        self.layers.append(nn.Linear(current_dim, dim_out))

    def forward(self, x):
        """
        [수정]
        MLP의 최종 출력(logits/z)과 함께
        중간 활성화 레이어의 출력(features)을 리스트로 반환합니다.
        """
        intermediate_features = []
        for layer in self.layers:
            x = layer(x)
            
            # ❗️ 활성화 함수(ReLU)의 출력을 '중간 특징'으로 간주하여 저장
            if isinstance(layer, nn.ReLU):
                intermediate_features.append(x)
        
        # final_output: 마지막 Linear 레이어의 출력
        # intermediate_features: 중간 ReLU 레이어들의 출력 리스트
        return x, intermediate_features

# -----------------------------------------------------------------
# 2. [수정] TeacherNet: `all_features` 딕셔너리 반환
# -----------------------------------------------------------------
class TeacherNet(nn.Module):
    def __init__(self, dim_A, dim_B, dim_y, num_classes, 
                 z_dim_A, z_dim_B, encA_hidden, encB_hidden, 
                 clf_hidden, p_drop, use_layernorm):
        super().__init__()
        # Encoder A: (dim_A) -> (z_dim_A)
        self.encoder_A = MLP(dim_A, z_dim_A, encA_hidden, p_drop, use_layernorm)
        # Encoder B: (dim_B) -> (z_dim_B)
        self.encoder_B = MLP(dim_B, z_dim_B, encB_hidden, p_drop, use_layernorm)
        # Classifier: (z_dim_A + z_dim_B) -> (dim_y * num_classes)
        self.classifier = MLP(z_dim_A + z_dim_B, dim_y * num_classes, clf_hidden, p_drop, use_layernorm)

    def forward(self, a, b):
        """
        [수정]
        MLP가 (output, features)를 반환하므로, 이를 모두 받아 처리
        """
        # 1. Encoders
        z_a, feats_A = self.encoder_A(a)
        z_b, feats_B = self.encoder_B(b)
        
        # 2. Classifier
        z_ab = torch.cat([z_a, z_b], dim=1)
        y_logits, feats_C = self.classifier(z_ab)
        
        # ❗️ 모든 중간 특징들을 딕셔너리로 묶어 반환
        all_features = {
            'encA': feats_A, # Encoder A의 중간 특징들
            'encB': feats_B,
            'clf': feats_C
        }
        
        # [수정] 반환값 변경 (z_b는 Student가 안 쓰므로 all_features를 대신 줌)
        return y_logits, z_a, all_features

# -----------------------------------------------------------------
# 3. [수정] StudentNet: `all_features` 딕셔너리 반환
# -----------------------------------------------------------------
class StudentNet(nn.Module):
    def __init__(self, dim_A, dim_y, num_classes, 
                 z_dim, enc_hidden, clf_hidden, 
                 p_drop, use_layernorm):
        super().__init__()
        # Encoder: (dim_A) -> (z_dim)
        # (Teacher의 encA_hidden과 동일한 구조를 사용해야 함)
        self.encoder = MLP(dim_A, z_dim, enc_hidden, p_drop, use_layernorm)
        # Classifier: (z_dim) -> (dim_y * num_classes)
        self.classifier = MLP(z_dim, dim_y * num_classes, clf_hidden, p_drop, use_layernorm)

    def forward(self, a):
        """
        [수정]
        Teacher와 마찬가지로 (output, features)를 받아 처리
        """
        # 1. Encoder
        z_a, feats_A = self.encoder(a)
        
        # 2. Classifier
        y_logits, feats_C = self.classifier(z_a)
        
        # ❗️ 모든 중간 특징들을 딕셔너리로 묶어 반환
        all_features = {
            'encA': feats_A, # Encoder A의 중간 특징들
            'clf': feats_C
        }
        
        # [수정] 반환값 변경 (2개 -> 3개)
        return y_logits, z_a, all_features