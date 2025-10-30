import os

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import OwlViTForObjectDetection, OwlViTProcessor


# ------------------------------------------------------------
# 1️⃣ OWL-ViT: Text-conditioned object detection
# ------------------------------------------------------------
def detect_animals_with_owlvit(image_path, prompts=None):
    if prompts is None:
        prompts = [
            "an animal", "a cat", "a dog",
            "a small cat", "a close-up of a dog",
            "a pet animal"
        ]

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, threshold=0.05, target_sizes=[image.size]
    )[0]

    scores = np.array(results["scores"])
    boxes = results["boxes"]
    labels = results["labels"]

    if len(scores) == 0:
        print("❌ No detections found.")
        return image, []

    # adaptive threshold: 상위 30% (낮을수록 많이 남김)
    adaptive_thresh = max(0.05, np.percentile(scores, 70))
    keep = scores >= adaptive_thresh

    boxes = boxes[keep]
    scores = scores[keep]
    labels = [prompts[i] for i in labels[keep]]

    print(f"✅ Adaptive threshold = {adaptive_thresh:.2f}, detections = {len(boxes)}")
    for lbl, sc, box in zip(labels, scores, boxes):
        print(f"{lbl:<15} | score={sc:.2f} | box={box.tolist()}")

    return image, boxes


# ------------------------------------------------------------
# 2️⃣ SAM: refine bounding boxes into precise masks
# ------------------------------------------------------------
def refine_with_sam(image, boxes, sam_checkpoint="data/sam_vit_h_4b8939.pth"):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    image_np = np.array(image)
    predictor.set_image(image_np)

    masks = []
    for box in boxes:
        box_np = np.array([box.tolist()])
        mask, _, _ = predictor.predict(box=box_np)
        masks.append(mask[0])

    if not masks:
        print("❌ No mask generated.")
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    combined = np.clip(np.sum(np.stack(masks), axis=0), 0, 1)
    mask_img = (combined * 255).astype(np.uint8)
    return mask_img


# ------------------------------------------------------------
# 3️⃣ 통합 실행 함수
# ------------------------------------------------------------
def create_mask_owlvit_sam(image_path, output_dir="data"):
    image, boxes = detect_animals_with_owlvit(image_path)
    boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes  # ← 추가

    if len(boxes) == 0:
        print("❌ No detections after thresholding.")
        return None

    mask_img = refine_with_sam(image, boxes)
    out_path = os.path.join(
        output_dir, os.path.basename(image_path).replace(".jpg", "_owlvit_sam_mask.png")
    )
    Image.fromarray(mask_img).save(out_path)
    print(f"🎯 Mask saved to {out_path}")
    return out_path

# ------------------------------------------------------------
if __name__ == "__main__":
    create_mask_owlvit_sam("data/2.jpg")
