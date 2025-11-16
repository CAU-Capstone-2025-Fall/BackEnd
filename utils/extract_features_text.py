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
import json
import re

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
    "아래 동물 사진을 보고, 겉모습의 특징을 한국어로 JSON 형태로 추출해줘.\n"
    "JSON에는 반드시 아래의 n개 항목이 포함되어야 해:\n"
    "1. main_color: '가장 두드러지는 털 색'\n"
    "2. sub_color: '두 번째로 두드러지는 털 색'\n"
    "3. fur_pattern: '털 무늬(자연적인 패턴이나 명확하게 보이는 특징만 기술)'\n"
    "4. fur_length: '털 길이(짧음/중간/김 중 하나)'\n"
    "5. fur_texture: '털 질감(부드러움/뻣뻣함 등)'\n"
    "6. rough_size: '대략적인 체구(작음/중간/큼 중 하나)'\n"
    "7. eye_shape: '눈 모양, 형태, 크기등의 특징'\n"
    "8. ear_shape: '귀 모양, 형태, 크기등의 특징'\n"
    "9. tail_shape: '꼬리 모양, 형태, 크기등의 특징'\n"
    "10. noticeable_features: '눈/귀/꼬리/체형 외에 명확한 기타 외형적 특징' \n"
    "11. health_impression: '사진상 가능한 건강에 대한 인상. 먼지, 진흙, 물기, 빗물 자국 등의 외부 요인은 제외' \n"
    "각 항목의 값은 한두 문장으로 서술하되, 문장 끝에 마침표를 붙여줘.\n"
    "모든 항목은 자연적이고 명확하게 보이는 특징만 기술하고, 추측이나 단정은 하지 마. 추정이 필요한 경우 ‘추정’, ‘가능성’ 등의 표현을 사용해.\n"
    """예시: {\"main_color\": \"흰 색.\", \"sub_color\": \"검은 색.\", \"fur_pattern\": \"검은 반점.\", \"fur_length\": \"짧음.\", \"fur_texture\": \"부드러움.\",
      \"rough_size\": \"중간 크기로 추정됨.\", \"eye_shape\": \"갈색 둥근 눈.\", \"ear_shape\": \"뾰족한 귀.\", \"tail_shape\": \"긴 꼬리.\", 
      \"noticeable_features\": null, \"health_impression\": \"겉보기에는 건강해 보이나, 자세한 상태는 알 수 없음.\"}"""
)

def try_https(u: str) -> str:
    return ("https://" + u[len("http://"):]) if u.startswith("http://") else u

def fetch_to_data_url(url: str) -> str:
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
    return " ".join(line.strip() for line in (text or "").splitlines()).strip()

def build_input_for_image(data_url: str):
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": TEXT_PROMPT},
            {"type": "input_image", "image_url": data_url},
        ],
    }]

def extract_json_like(text: str):
    # 1. 코드블록 형태 ```json ... ``` 또는 ``` ... ```
    m = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 2. 그냥 중괄호로 시작하는 가장 큰 JSON을 탐색 (중괄호 pair 최대 매칭)
    # 최초 { ... } 형태
    m = re.search(r"({[\s\S]*})", text)
    if m:
        return m.group(1)

    return None

def sanitize_json_string(json_str: str):
    """
    json.loads에서 흔히 실패하는 문제상황을 보정(큰따옴표/작은따옴표, trailing commas 등)
    단, 완벽하지는 않음(정규화 실패시 raw_string 저장)
    """
    # 작은따옴표 -> 큰따옴표로 변경 (단, 값에 ' 가 있을 순 있음, 완벽X)
    json_str = re.sub(r"'", r'"', json_str)
    # 마지막 쉼표 제거: {"a": 1, } 이런 형태
    json_str = re.sub(r",(\s*[\}\]])", r"\1", json_str)
    return json_str

def call_vision_json(data_url: str) -> dict:
    """OpenAI Vision 호출 후 JSON 파싱 강화"""
    resp = client_ai.responses.create(
        model=MODEL,
        input=build_input_for_image(data_url),
    )
    output = one_paragraph(resp.output_text)

    json_like = extract_json_like(output)
    if json_like:
        json_like = sanitize_json_string(json_like)
        try:
            result = json.loads(json_like)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # 혹시 그냥 전체가 잘 구성된 JSON일수도
    try:
        result = json.loads(output)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {"parse_error": output}

def try_parse_existing_feature(val) -> dict:
    """기존 추출된 특징 문자열을 json(dict)으로 변환"""
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or not val.strip():
        return {}
    try:
        result = json.loads(val)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"raw_string": val}

def is_empty_extracted_feature(doc: dict) -> bool:
    ef = doc.get("extractedFeature")
    if isinstance(ef, dict):
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in ef.values()):
            return True
        return False
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
    overwritten = 0

    # collection.update_many({}, {"$unset": {"extractedFeature": ""}})

    for doc in collection.find():
        try:
            ef_orig = doc.get("extractedFeature")

            if not is_empty_extracted_feature(doc):
                ef_json = try_parse_existing_feature(ef_orig)
                if not (isinstance(ef_orig, dict) and len(ef_orig) > 0):
                    collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"extractedFeature": ef_json}}
                    )
                    overwritten += 1
                skipped += 1
                continue

            img_url = doc.get("popfile1")
            if not img_url or not isinstance(img_url, str) or not img_url.strip():
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": {
                        "error": "이미지 없음(분석 생략)."
                    }}}
                )
                skipped += 1
                continue

            try:
                data_url = fetch_to_data_url(img_url)
            except Exception as e:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": {
                        "error": f"이미지 다운로드 실패: {str(e)[:150]}"
                    }}
                })
                failed += 1
                continue

            try:
                summary_json = call_vision_json(data_url)
            except Exception as e:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"extractedFeature": {
                        "error": f"분석 오류: {type(e).__name__}: {str(e)[:150]}"
                    }}
                })
                failed += 1
                continue

            if not summary_json or not isinstance(summary_json, dict):
                summary_json = {"error": "분석 결과가 비어 있습니다."}

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"extractedFeature": summary_json}}
            )
            processed += 1
            print(f"[OK] {doc.get('desertionNo')} 저장 완료")

            time.sleep(SLEEP_BETWEEN)

        except Exception as e:
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"extractedFeature": {
                    "error": f"처리 오류: {type(e).__name__}: {str(e)[:150]}"
                }}
            })
            failed += 1
            continue

    print(f"완료: 처리 {processed}, 건너뜀 {skipped}, 기존 문자열 덮어씀 {overwritten}, 실패 {failed}")