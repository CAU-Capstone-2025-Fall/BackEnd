import os
import time
import base64
import mimetypes
import requests
from io import BytesIO
from dotenv import load_dotenv
from pymongo import MongoClient
from PIL import Image
from openai import OpenAI

# -------------------- 환경 --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI    = os.getenv("MONGODB_URI")
MODEL          = "gpt-4.1-mini"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 가 필요합니다.")

client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]

# -------------------- 설정 --------------------
UA             = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
MAX_SIDE       = 1280
JPEG_QUALITY   = 85
SLEEP_BETWEEN  = 0.8

# -------------------- 프롬프트 --------------------
TEXT_PROMPT = (
    "아래 동물 사진을 보고, 겉모습의 특징을 한국어 한 문단(250~450자)으로 정리해줘.\n"
    "불릿, 줄바꿈 없이 문장형 서술만 작성하고, 추측은 ‘추정’, ‘가능성’ 같은 표현으로 완곡하게 써.\n"
    "포함: 1) 털 색/무늬, 2) 대략적 체구(작음/중간/큼), 3) 모질(길이/결감), "
    "4) 눈/귀/체형 등 눈에 띄는 특징, 5) 사진만으로 가능한 건강 인상(‘추정’, ‘가능성’ 등 보수적 표현).\n"
    "주의:\n"
    "- 일시적인 먼지, 진흙, 물기, 빗물 자국 등 **씻거나 닦으면 사라질 수 있는 외부 오염**은 건강 인상에서 제외한다.\n"
    "- 털의 색상·무늬는 **자연적인 패턴이나 명확히 관찰 가능한 특징만** 기술한다.\n"
    "- 품종, 나이, 질병은 단정 금지. 보이는 사실 위주로 서술하고 과장하지 않는다."
)

# -------------------- 유틸 --------------------
def try_https(u: str) -> str:
    """http:// → https:// 자동 변환"""
    return ("https://" + u[len("http://"):]) if u.startswith("http://") else u

def fetch_to_data_url(url: str) -> str:
    """popfile1 URL을 다운로드 → 리사이즈 → base64 data URL로 변환"""
    u = try_https(url)
    headers = {"User-Agent": UA, "Referer": url}
    r = requests.get(u, headers=headers, timeout=12, stream=True, allow_redirects=True)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(MAX_SIDE)
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)

    bio = BytesIO()
    img.save(bio, format="JPEG", quality=JPEG_QUALITY)
    b = bio.getvalue()

    ctype = "image/jpeg"
    guess = mimetypes.guess_type(u)[0]
    if guess and guess.startswith("image/"):
        ctype = guess

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{ctype};base64,{b64}"

def one_paragraph(text: str) -> str:
    """여러 줄 텍스트를 한 문단으로 변환"""
    return " ".join(line.strip() for line in (text or "").splitlines()).strip()

def build_input_for_image(data_url: str):
    """OpenAI Responses API 입력 형식 구성"""
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": TEXT_PROMPT},
            {"type": "input_image", "image_url": data_url},
        ],
    }]

def call_vision_text(data_url: str) -> str:
    """OpenAI Vision 호출"""
    resp = client_ai.responses.create(
        model=MODEL,
        input=build_input_for_image(data_url),
    )
    return one_paragraph(resp.output_text)

def is_empty_extracted_feature(doc: dict) -> bool:
    ef = doc.get("extractedFeature")
    return (ef is None) or (isinstance(ef, str) and ef.strip() == "")

# -------------------- 메인 --------------------
if __name__ == "__main__":
    try:
        collection.create_index("desertionNo", unique=True)
        print("DB 연결 및 인덱스 확인 완료")
    except Exception as e:
        print("DB 연결/인덱스 오류:", e)

    processed = 0
    skipped = 0
    failed = 0

    for doc in collection.find():
        try:
            if not is_empty_extracted_feature(doc):
                skipped += 1
                continue

            img_url = doc.get("popfile1")
            if not img_url or not isinstance(img_url, str) or not img_url.strip():
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": "이미지 없음(분석 생략)."}}
                )
                skipped += 1
                continue

            try:
                data_url = fetch_to_data_url(img_url)
            except Exception as e:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": f"이미지 다운로드 실패: {str(e)[:150]}"}}
                )
                failed += 1
                continue

            try:
                summary = call_vision_text(data_url)
            except Exception as e:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": f"분석 오류: {type(e).__name__}: {str(e)[:150]}"}}
                )
                failed += 1
                continue

            if not summary:
                summary = "분석 결과가 비어 있습니다."

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"extractedFeature": summary}}
            )
            processed += 1
            print(f"[OK] {doc.get('desertionNo')} 저장 완료")

            # 잠시 쉬기(레이트리밋 완화)
            time.sleep(SLEEP_BETWEEN)

        except Exception as e:
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"extractedFeature": f"처리 오류: {type(e).__name__}: {str(e)[:150]}"}}
            )
            failed += 1
            continue

    print(f"완료: 처리 {processed}, 건너뜀 {skipped}, 실패 {failed}")
