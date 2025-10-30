import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from dotenv import load_dotenv  # ✅ 추가
from openai import OpenAI
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# ✅ .env 파일 로드
load_dotenv()

# ---------------------------------------
# 1️⃣ CLIPSeg 마스크 생성
# ---------------------------------------
def generate_mask(image_path, prompt="a photo of an animal", threshold=0.5):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    mask_logits = outputs.logits.mean(0).squeeze()
    mask = torch.sigmoid(mask_logits).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask = np.power(mask, 0.7)
    binary_mask = (mask > threshold).astype(np.uint8)

    mask_img = Image.fromarray((binary_mask * 255).astype("uint8")).resize(image.size)
    mask_path = image_path.replace(".jpg", "_mask.png")
    mask_img.save(mask_path)
    print(f"✅ Binary mask saved to {mask_path}")
    return mask_path


def dalle3_inpaint(image_path, mask_path, prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in environment")

    headers = {"Authorization": f"Bearer {api_key}"}

    # ✅ ensure both are identical size + RGBA
    img = Image.open(image_path).convert("RGBA").resize((1024, 1024))
    mask = Image.open(mask_path).convert("RGBA").resize(img.size)

    # ✅ temporary aligned copies
    img.save("temp_image.png")
    mask.save("temp_mask.png")

    files = {
        "image": ("image.png", open("temp_image.png", "rb"), "image/png"),
        "mask": ("mask.png", open("temp_mask.png", "rb"), "image/png"),
    }
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    resp = requests.post("https://api.openai.com/v1/images/edits", headers=headers, files=files, data=data)
    if resp.status_code != 200:
        raise RuntimeError(f"API Error: {resp.text}")

    result = resp.json()["data"][0]
    if "b64_json" in result:
        image_bytes = base64.b64decode(result["b64_json"])
        image = Image.open(io.BytesIO(image_bytes))
        out_path = image_path.replace(".jpg", "_inpainted.png")
        image.save(out_path)
        print(f"🎨 Inpainted image saved to {out_path}")
        return image
    else:
        raise ValueError("❌ No base64 image data returned")

# ---------------------------------------
# 3️⃣ 실행
# ---------------------------------------
if __name__ == "__main__":
    image_path = "data/2.jpg"
    mask_path = generate_mask(image_path)

    prompt = (
        "Please gently clean this animal while preserving its real-world appearance. Remove only visible dirt, stains, and foreign matter from the fur — without altering the natural fur color, texture, facial structure, body shape, or any breed-specific features. Keep medical conditions or unique physical traits (such as skin marks or scars) clearly visible and untouched. Do not generate a new animal or change the expression. Maintain the current setting and background as is, only enhancing overall image clarity, sharpness, and lighting to better reflect the original scene. Keep all colors and tones realistic, natural, and consistent with real-life lighting — no artificial or exaggerated color grading. The final image should look like an unedited high-quality photo of the same animal in the same environment."
    )

    result_img = dalle3_inpaint(image_path, mask_path, prompt)

    orig = Image.open(image_path)
    mask = Image.open(mask_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(result_img)
    axes[2].set_title("Inpainted Result")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
