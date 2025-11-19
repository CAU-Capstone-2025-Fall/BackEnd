# routers/gptgenerator.py

import base64
import os
from io import BytesIO

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

router = APIRouter(prefix="/gpt-image", tags=["GPTImage"])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_and_resize_image(img_bytes: bytes, size=(1024, 1024)):
    print("[DEBUG] load_and_resize_image: received image bytes =", len(img_bytes))

    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print("[ERROR] PIL cannot open image:", e)
        raise

    print("[DEBUG] Original size:", image.size)
    image = image.resize(size, Image.LANCZOS)
    print("[DEBUG] Resized size:", image.size)

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


@router.post("/clean")
async def clean_image(file: UploadFile = File(...)):
    print("\n======================")
    print("[API CALL] /gpt-image/clean")
    print("======================")

    if not file:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")

    FIXED_PROMPT = (
        "Please gently clean this animal while preserving its real-world appearance. "
        "Remove only visible dirt, stains, and foreign matter from the fur — without altering "
        "the natural fur color, texture, facial structure, body shape, or any breed-specific features. "
        "Keep medical conditions or unique physical traits (such as skin marks or scars) clearly visible and untouched. "
        "Do not generate a new animal or change the expression. "
        "Maintain the current setting and background as is, only enhancing overall image clarity, sharpness, and lighting "
        "to better reflect the original scene. "
        "Keep all colors and tones realistic, natural, and consistent with real-life lighting — "
        "no artificial or exaggerated color grading. "
        "The final image should look like an unedited high-quality photo of the same animal in the same environment."
    )

    try:
        raw_bytes = await file.read()
        print("[DEBUG] Uploaded file:", file.filename)
        print("[DEBUG] Raw bytes =", len(raw_bytes))

        resized_buf = load_and_resize_image(raw_bytes)

        # ============================================
        # 🔥 REST API로 /v1/images/edits 직접 호출
        # ============================================
        url = "https://api.openai.com/v1/images/edits"

        files = {
            "image": ("input.png", resized_buf, "image/png"),   # 반드시 이렇게 해야 함
        }

        data = {
            "model": "gpt-image-1",
            "prompt": FIXED_PROMPT,
            "size": "1024x1024",
            "n": 1,
            "input_fidelity": "high",
            "quality": "high",
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        print("[DEBUG] Sending POST to /v1/images/edits...")
        response = requests.post(url, headers=headers, files=files, data=data)

        print("[DEBUG] Response status:", response.status_code)

        if response.status_code != 200:
            print("[RAW ERROR]", response.text)
            raise Exception(response.text)

        resp_json = response.json()
        print("[RAW RESPONSE]", resp_json)

        b64 = resp_json["data"][0]["b64_json"]

        return {
            "success": True,
            "image_base64": b64,
            "prompt_used": FIXED_PROMPT
        }

    except Exception as e:
        print("\n[EXCEPTION] =======================")
        print(e)
        print("===================================\n")
        raise HTTPException(status_code=500, detail=str(e))
