import base64
import os
import uuid
from io import BytesIO

import firebase_admin
import requests
from dotenv import load_dotenv
from firebase_admin import credentials, storage
from pymongo import MongoClient

# ==========================
# 환경변수 로드
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")

# ==========================
# MongoDB 연결
# ==========================
client = MongoClient(MONGO_URI)
db = client["testdb"]
animals_col = db["abandoned_animals"]

print("🔥 MongoDB 연결 완료")
print("🔥 대상 컬렉션: testdb.abandoned_animals")


# ==========================
# Firebase 초기화
# ==========================
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred, {
    "storageBucket": "capstone-366a6.firebasestorage.app"
    })

bucket = storage.bucket()

def upload_to_firebase(base64_img: str, filename_prefix: str) -> str:
    """base64 이미지를 Firebase Storage에 업로드하고 public URL 반환"""
    try:
        img_bytes = base64.b64decode(base64_img)
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
        blob = bucket.blob(filename)
        blob.upload_from_string(img_bytes, content_type="image/png")
        blob.make_public()

        print(f"📤 Firebase 업로드 성공: {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print("❌ Firebase 업로드 실패:", e)
        return None


# ==========================
# GPT Image Edit 호출
# ==========================
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
        raise Exception(resp.text)

    return resp.json()["data"][0]["b64_json"]


# ==========================
# 메인 처리 루프
# ==========================
def process_animals():
    print("\n🚀 improve != '0' 동물 이미지 생성 시작\n")

    # improve가 0이 아닌 문서만 선택
    targets = list(animals_col.find({"improve": {"$in": ["1", "2"]}}))

    print(f"🎯 대상 동물 수: {len(targets)}")

    for animal in targets:
        try:
            desertion_no = animal["desertionNo"]
            popfile1 = animal.get("popfile1")

            print(f"\n==============================")
            print(f"🐾 처리중: {desertion_no}")
            print(f"📥 원본 이미지 URL: {popfile1}")

            # 1) 원본 이미지 다운로드
            img_bytes = requests.get(popfile1).content

            # 2) GPT 이미지 생성
            print("🎨 GPT 이미지 생성 중…")
            b64_img = generate_clean_image(img_bytes)

            # 3) Firebase 업로드
            fb_url = upload_to_firebase(b64_img, desertion_no)

            if not fb_url:
                print("❌ Firebase 업로드 실패 → 스킵")
                continue

            # 4) DB 업데이트 (createdImg 덮어쓰기)
            animals_col.update_one(
                {"desertionNo": desertion_no},
                {"$set": {"createdImg": fb_url}}
            )

            print(f"✅ 완료: {desertion_no}")
            print(f"   → createdImg 저장됨")

        except Exception as e:
            print(f"❌ ERROR({desertion_no}):", e)
            continue


if __name__ == "__main__":
    process_animals()
    print("\n🎉 모든 작업 완료!\n")
