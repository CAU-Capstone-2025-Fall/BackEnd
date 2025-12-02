# routers/gptgenerator.py

import base64
import os
import uuid
from io import BytesIO

import firebase_admin
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from firebase_admin import credentials, storage
from pymongo import MongoClient

# ========================================
# FastAPI 라우터
# ========================================
router = APIRouter(prefix="/gpt-image", tags=["GPTImage"])

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")

# ========================================
# MongoDB 연결
# ========================================
mongo = MongoClient(MONGO_URI)
db = mongo["testdb"]
animals_col = db["abandoned_animals"]

print("🔥 MongoDB 연결: testdb.abandoned_animals")


# ========================================
# Firebase 초기화
# ========================================
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "capstone-366a6.firebasestorage.app"
    })

bucket = storage.bucket()


# ========================================
# Firebase 업로드 함수
# ========================================
def upload_to_firebase(base64_img: str, filename_prefix: str) -> str:
    try:
        img_bytes = base64.b64decode(base64_img)
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
        blob = bucket.blob(filename)

        blob.upload_from_string(img_bytes, content_type="image/png")
        blob.make_public()

        print(f"📤 Firebase 업로드 성공 → {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print("❌ Firebase 업로드 실패:", e)
        return None


# ========================================
# GPT 프롬프트
# ========================================
GPT_PROMPT = (
    "Please clean this animal while strictly preserving its real-world appearance. "
    "Remove only visible dirt, stains, mud, dust, and foreign particles from the fur. "
    "Do NOT modify: "
    "- the natural fur color, shade, tone, or brightness "
    "- the lighting, exposure, white balance, or color temperature "
    "- the contrast, saturation, vibrance, or overall color grade "
    "- the facial structure, body shape, eye color, or any breed-specific traits "
    "- shadows, highlights, or natural lighting direction in the scene "
    "The cleaned fur must retain the exact same original color and darkness. "
    "Do NOT brighten the image, do NOT whiten the fur, and do NOT smooth excessive texture. "
    "Avoid all aesthetic enhancement, glamorization, upscaling effects, or style changes. "
    "Keep all medical conditions, scars, markings, and unique physical features clearly visible and unchanged. "
    "Do not replace the animal, do not redraw the face, and do not create new details. "
    "Enhance overall clarity only in a subtle way, without altering the color or tone of any region. "
    "Preserve the original background, perspective, depth, and environment exactly as captured. "
    "The final image should look like the same photo—just naturally cleaner, not edited."
)


# ========================================
# GPT 이미지 생성
# ========================================
def generate_clean_image(img_bytes: bytes) -> str:
    url = "https://api.openai.com/v1/images/edits"

    files = {
        "image": ("input.png", img_bytes, "image/png"),
    }

    data = {
        "model": "gpt-image-1",
        "prompt": GPT_PROMPT,
        "size": "1024x1024",
        "n": 1,
        "input_fidelity": "high",
        "quality": "high",
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    resp = requests.post(url, headers=headers, files=files, data=data)

    if resp.status_code != 200:
        print("🔥 GPT ERROR:", resp.text)
        raise Exception(resp.text)

    return resp.json()["data"][0]["b64_json"]
@router.post("/clean")
async def clean_image(
    desertionNo: str = Form(...)
):
    """
    desertionNo 만 입력하면
    DB에서 popfile1 자동으로 읽어서 처리
    """
    try:
        print(f"\n🔔 /gpt-image/clean 호출 → desertionNo={desertionNo}")

        # 1. DB 존재 확인
        animal = animals_col.find_one({"desertionNo": desertionNo})
        if not animal:
            raise HTTPException(404, f"동물({desertionNo})을 DB에서 찾을 수 없음")

        # 2. popfile1 가져오기
        img_url = animal.get("popfile1")
        if not img_url:
            raise HTTPException(400, f"동물 {desertionNo} 의 popfile1 이 존재하지 않음")

        print(f"🌐 popfile1 이미지 URL: {img_url}")

        # 3. URL에서 이미지 다운로드
        resp = requests.get(img_url)
        if resp.status_code != 200:
            raise HTTPException(400, f"이미지 다운로드 실패: {img_url}")

        img_bytes = resp.content
        print(f"📥 이미지 다운로드 성공: {len(img_bytes)} bytes")

        # 4. GPT 이미지 생성
        print("🎨 GPT 이미지 생성 중…")
        b64_img = generate_clean_image(img_bytes)

        # 5. Firebase 업로드
        fb_url = upload_to_firebase(b64_img, desertionNo)
        if not fb_url:
            raise HTTPException(500, "Firebase 업로드 실패")

        # 6. DB 업데이트
        animals_col.update_one(
            {"desertionNo": desertionNo},
            {"$set": {"createdImg": fb_url, "improve": "1"}}
        )

        print("✅ DB 업데이트 완료")

        return {
            "success": True,
            "desertionNo": desertionNo,
            "createdImg": fb_url,
            "message": "popfile1 기반 이미지 클린 완료"
        }

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
