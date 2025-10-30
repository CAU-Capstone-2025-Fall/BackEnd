import base64
import io
import os

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from firebase_admin import credentials, initialize_app, storage
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from pymongo import MongoClient
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# ----------------------------------------------------
# 1️⃣ 초기 설정
# ----------------------------------------------------
load_dotenv()

# Firebase 초기화
cred = credentials.Certificate("/home/ec2-user/BackEnd/firebase-key.json")
#cred = credentials.Certificate("firebase-key.json") # 로컬 테스트용
initialize_app(cred, {"storageBucket": os.getenv("FIREBASE_BUCKET")})
bucket = storage.bucket()

# MongoDB 연결
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["testdb"]
collection = db["abandoned_animals"]

app = FastAPI(title="Animal Cleaner API")
router = APIRouter(prefix="/image", tags=["image"])

# OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client_oa = OpenAI(api_key=api_key)


class CleanRequest(BaseModel):
    desertionNo: str  # ✅ 이제 noticeNo 대신 desertionNo 사용


# ----------------------------------------------------
# Firebase 업로드 유틸
# ----------------------------------------------------
def upload_to_firebase(image: Image.Image, filename: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    blob = bucket.blob(f"cleaned/{filename}")
    blob.upload_from_file(buf, content_type="image/png")
    blob.make_public()

    return blob.public_url


# ----------------------------------------------------
# CLIPSeg 마스크 생성
# ----------------------------------------------------
def generate_mask(image: Image.Image, threshold=0.5):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(
        text=["a photo of an animal"],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    mask_logits = outputs.logits.mean(0).squeeze()
    mask = torch.sigmoid(mask_logits).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask = np.power(mask, 0.7)
    binary_mask = (mask > threshold).astype(np.uint8)

    mask_img = Image.fromarray((binary_mask * 255).astype("uint8")).resize(image.size)
    return mask_img


# ----------------------------------------------------
# DALL·E 3 인페인팅
# ----------------------------------------------------
def dalle3_inpaint(image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
    headers = {"Authorization": f"Bearer {api_key}"}

    image = image.convert("RGBA").resize((1024, 1024))
    mask = mask.convert("RGBA").resize((1024, 1024))

    image.save("temp_image.png")
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
        raise HTTPException(status_code=400, detail=resp.text)

    result = resp.json()["data"][0]
    image_bytes = base64.b64decode(result["b64_json"])
    return Image.open(io.BytesIO(image_bytes))


# ----------------------------------------------------
# 이미지 자동 보정 (desertionNo 기반)
# ----------------------------------------------------
@router.post("/auto-clean")
async def auto_clean(req: CleanRequest):
    # 1️⃣ MongoDB에서 desertionNo로 문서 찾기
    animal = collection.find_one({"desertionNo": req.desertionNo})
    if not animal:
        raise HTTPException(status_code=404, detail=f"Animal with desertionNo '{req.desertionNo}' not found")

    image_url = animal.get("popfile1")
    if not image_url:
        raise HTTPException(status_code=404, detail="popfile1 (original image URL) not found")

    # 2️⃣ 이미지 다운로드
    resp = requests.get(image_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Image not found at popfile1 URL")

    image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 3️⃣ 마스크 생성
    mask_img = generate_mask(image)

    # 4️⃣ DALL·E 3 보정
    prompt = (
        "Please gently clean this animal while preserving its real-world appearance. "
        "Remove only visible dirt, stains, and foreign matter from the fur — without altering "
        "the natural fur color, texture, facial structure, body shape, or any breed-specific features. "
        "Keep medical conditions or unique physical traits (such as skin marks or scars) clearly visible "
        "and untouched. Do not generate a new animal or change the expression. Maintain the current setting "
        "and background as is, only enhancing overall image clarity, sharpness, and lighting realistically."
    )

    result_img = dalle3_inpaint(image, mask_img, prompt)

    # 5️⃣ Firebase 업로드
    clean_url = upload_to_firebase(result_img, f"{req.desertionNo}_cleaned.png")

    # 6️⃣ MongoDB 업데이트
    collection.update_one(
        {"desertionNo": req.desertionNo},
        {"$set": {"createdImg": clean_url}},
    )

    return {
        "desertionNo": req.desertionNo,
        "original_url": image_url,
        "createdImg": clean_url,
        "message": "Image cleaned and saved successfully."
    }


app.include_router(router)
