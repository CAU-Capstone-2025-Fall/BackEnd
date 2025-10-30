import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoModel, AutoProcessor


def get_similarity_map(image_path, text_prompt="dog, cat, animal"):
    model_id = "google/siglip-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to("cpu")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=[text_prompt], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        vision_feats = outputs.vision_model_output.last_hidden_state[0]
        text_feat = outputs.text_embeds[0]
        sim = (vision_feats @ text_feat).squeeze()
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        h = w = int(np.sqrt(sim.shape[0]))
        sim_map = sim.reshape(h, w).numpy()
    return np.array(image), sim_map

def extract_boxes_from_heatmap(sim_map, thresh=0.6):
    mask = (sim_map > thresh).astype(np.uint8) * 255
    mask = cv2.resize(mask, (512, 512))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return [[x, y, x + w, y + h] for (x, y, w, h) in boxes]

def refine_with_sam(image_np, boxes, sam_checkpoint="data/sam_vit_h_4b8939.pth"):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image_np)
    masks = []
    for box in boxes:
        mask, _, _ = predictor.predict(box=np.array([box]))
        masks.append(mask[0])
    combined = np.clip(np.sum(np.stack(masks), axis=0), 0, 1)
    return (combined * 255).astype(np.uint8)

image_np, sim_map = get_similarity_map("data/2.jpg")
boxes = extract_boxes_from_heatmap(sim_map, thresh=0.6)
mask_img = refine_with_sam(image_np, boxes)
Image.fromarray(mask_img).save("data/2_siglip_sam_mask.png")
