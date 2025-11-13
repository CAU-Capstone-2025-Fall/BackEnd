# ============================================================
# run_distill.py — (회귀: Y_4labels) Teacher–Student Distillation
# ============================================================

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from dataio import load_config, load_inputs_and_labels, resolve_device
from models import StudentNet, TeacherNet
from trainer import train_student_distill, train_teacher


def main():
    base = Path(__file__).resolve().parent

    # ── 1) 설정 로드
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        cfg = load_config(str(cfg_path))
    else:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    DEVICE = resolve_device(cfg.get("device", "auto"))
    torch.set_float32_matmul_precision("medium")

    # ── 2) 데이터 로드 (A_noex, B_clean, Y_4labels)
    A, B, Y = load_inputs_and_labels(cfg)
    print(f"✅ [LOAD COMPLETE] Shapes → A={np.shape(A)}, B={np.shape(B)}, Y={np.shape(Y)}")

    # Convert to tensors
    A = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    B = torch.tensor(B, dtype=torch.float32, device=DEVICE)
    
    # ❗️ [수정] Y를 'float' (회귀) 타입으로 로드
    Y = torch.tensor(Y, dtype=torch.float32, device=DEVICE) 

    # ❗️ [수정] dim_y는 Y의 컬럼 수 (예: 4)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1) # (N,) -> (N, 1)
    dim_y = Y.shape[1] 
    print(f"✅ [INFO] Target dim_y = {dim_y}")

    # ── 3) 모델/학습 설정 구성 (회귀용)
    # ❗️ [수정] 'num_classes' 키 제거
    teacher_model_cfg = {
        "z_dim_A": cfg["teacher_model"]["z_dim_A"],
        "z_dim_B": cfg["teacher_model"]["z_dim_B"],
        "encA_hidden": tuple(cfg["teacher_model"]["encA_hidden"]),
        "encB_hidden": tuple(cfg["teacher_model"]["encB_hidden"]),
        "clf_hidden": tuple(cfg["teacher_model"]["clf_hidden"]),
        "p_drop": cfg["teacher_model"]["p_drop"],
        "use_layernorm": cfg["teacher_model"].get("use_layernorm", False),
        "dim_A": A.shape[1], # ❗️ (추가) models.py가 필요로 할 경우 대비
        "dim_B": B.shape[1], # ❗️ (추가)
        "dim_y": dim_y,      # ❗️ (추가)
    }

    student_model_cfg = {
        "z_dim": cfg["student_model"]["z_dim"],
        "enc_hidden": tuple(cfg["student_model"]["enc_hidden"]),
        "clf_hidden": tuple(cfg["student_model"]["clf_hidden"]),
        "p_drop": cfg["student_model"]["p_drop"],
        "use_layernorm": cfg["student_model"].get("use_layernorm", False),
        "dim_A": A.shape[1], # ❗️ (추가)
        "dim_y": dim_y,      # ❗️ (추가)
    }
    
    # ❗️ (중요) 'num_classes'가 config.yaml에 있어도 무시하고 모델에 안 넘김
    teacher_model_cfg.pop("num_classes", None)
    student_model_cfg.pop("num_classes", None)


    teacher_train_cfg = {**cfg["teacher_train"], "device": DEVICE}
    student_train_cfg = {**cfg["student_train"], "device": DEVICE}

    # ── 4) Teacher 학습
    print("\n🚀 Training Teacher Network (Regression)...")
    teacher, teacher_hist = train_teacher(
        A, B, Y, dim_y,
        model_cfg=teacher_model_cfg,
        train_cfg=teacher_train_cfg,
        TeacherNet=TeacherNet,
        loss_weights=cfg.get("loss_weights", {})
    )

    # ── 5) Student(KD) 학습
    print("\n🎓 Training Student Network with KD (Regression)...")
    student, student_hist = train_student_distill(
        A, B, Y,
        teacher=teacher,
        dim_y=dim_y,
        model_cfg=student_model_cfg,
        train_cfg=student_train_cfg,
        StudentNet=StudentNet,
        kd_cfg=cfg.get("student_kd", {}),
        init_from_teacher=True
    )

    # ── 6) 미리보기 (Student는 A만 입력받음)
    preview_rows = int(cfg.get("inference", {}).get("preview_rows", 5))
    preview_rows = min(preview_rows, A.shape[0])

    student.eval()
    with torch.no_grad():
        A_preview = A[:preview_rows].to(DEVICE, dtype=torch.float32)
        out = student(A_preview)
        preds = out[0] if isinstance(out, tuple) else out
        preds_np = preds.detach().cpu().numpy()

    print("\n[Preview Predictions (Regression)]")
    print(preds_np)

    # ── 7) Teacher latent 품질 분석 및 T–S 정렬 평가
    try:
        from evaluation_utils import (
            evaluate_teacher_latent,
            plot_teacher_student_alignment,
        )
        
        os.makedirs(base / "analysis_results", exist_ok=True)
        analysis_dir = os.path.join(base, "analysis_results")

        print("\n[Auto-Eval] Running Teacher latent quality analysis...")
        evaluate_teacher_latent(
            teacher, A.cpu().numpy(), B.cpu().numpy(),
            save_dir=analysis_dir
        )

        print("\n[Auto-Eval] Running Teacher–Student alignment analysis...")
        plot_teacher_student_alignment(
            teacher, student,
            A.cpu().numpy(), B.cpu().numpy(),
            save_dir=analysis_dir
        )
    except ImportError:
        print("\n[WARN] 'evaluation_utils' not found. Skipping auto-evaluation.")
    except Exception as e:
        print(f"[WARN] Auto-evaluation skipped: {e}")

    # ── 8) 저장 위치 안내
    print("\n✅ All training completed successfully.")
    # ❗️ (수정) config.yaml에서 실제 체크포인트 파일명 참조
    teacher_ckpt_name = cfg["teacher_train"].get("ckpt_name", "teacher.pt")
    student_ckpt_name = cfg["student_train"].get("ckpt_name", "student.pt")
    
    print(f"   ├─ Teacher checkpoint: {teacher_train_cfg['ckpt_dir']}/{teacher_ckpt_name}")
    print(f"   └─ Student checkpoint: {student_train_cfg['ckpt_dir']}/{student_ckpt_name}")


if __name__ == "__main__":
    main()