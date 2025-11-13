import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# 모델 로드
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 이미지 로드
image = Image.open("data/1.jpg").convert("RGB")

# 프롬프트 (좋음 👍)
prompt = ["a photo of an animal"]

# padding, truncation 추가 👇
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

with torch.no_grad():
    outputs = model(**inputs)

# 여러 프롬프트 마스크 평균
mask_logits = outputs.logits.mean(0).squeeze()
mask = torch.sigmoid(mask_logits).numpy()

# normalize + 감마 보정 (대부분 다 검은 문제 해결)
mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
mask = np.power(mask, 0.7)

# 원본 크기로 resize 후 저장
mask_img = Image.fromarray((mask * 255).astype("uint8")).resize(image.size)
mask_img.save("data/1_clipseg_mask.png")
print("🎯 Mask saved to data/1_clipseg_mask.png")
