import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ------------------------------------------------------------
# 1️⃣ 모델 로드
# ------------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# ------------------------------------------------------------
# 2️⃣ 입력 이미지 + 텍스트
# ------------------------------------------------------------
image_path = "data/2.jpg"
image = Image.open(image_path).convert("RGB")
text_prompt = "cat, dog, animal"

inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# ------------------------------------------------------------
# 3️⃣ Forward pass (attention map 추출)
# ------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

print(f"decoder_attentions length: {len(outputs.decoder_attentions)}")

# 마지막 layer, 마지막 scale tensor만 선택
last_layer = outputs.decoder_attentions[-1]
while isinstance(last_layer, tuple):
    last_layer = last_layer[-1]

print(f"attention tensor shape: {last_layer.shape}")

# 평균 head attention
attn_map = last_layer.mean(dim=2).squeeze(0)  # [heads, 4, 4]
attn = attn_map.mean(dim=0).cpu().numpy()     # [4,4] 평균 head

# normalize
attn = (attn - attn.min()) / (attn.max() - attn.min())

# ------------------------------------------------------------
# 4️⃣ Heatmap 시각화
# ------------------------------------------------------------
attn_resized = cv2.resize(attn, image.size, interpolation=cv2.INTER_CUBIC)
heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
image_np = np.array(image)
overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

plt.imshow(overlay)
plt.axis("off")
plt.title(f"Grounding DINO Attention ('{text_prompt.split(',')[0].strip()}')")
plt.show()

out_path = image_path.replace(".jpg", "_groundingDINO_heatmap.png")
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"✅ Heatmap saved to {out_path}")
