import base64
import os
from io import BytesIO

import requests
from fastapi import APIRouter, FastAPI
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

# 환경 변수에서 API 키 읽기
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# FastAPI 앱
app = FastAPI(title="Image Edit Backend")

# 라우터 등록
router = APIRouter(prefix="/image", tags=["image"])

# 요청 스키마
class ImageEditRequest(BaseModel):
    prompt: str
    image_url: str


# URL 이미지를 PNG + Mask 로 변환
def url_to_png(image_url: str):
    resp = requests.get(image_url)
    if resp.status_code != 200:
        return None, None

    # 원본 RGBA 변환 및 리사이즈
    image = Image.open(BytesIO(resp.content)).convert("RGBA")
    image = image.resize((1024, 1024), Image.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # 마스크 전체 투명 (즉, 전체 영역 수정 가능)
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    buf_mask = BytesIO()
    mask.save(buf_mask, format="PNG")
    buf_mask.seek(0)

    return buf, buf_mask


@router.post("/edit")
async def edit_image(req: ImageEditRequest):
    """
    1. GPT로 프롬프트 정제
    2. DALL·E3 API(images/edits) 호출
    3. 결과 URL 또는 base64 반환
    """
    # 1) GPT-4o-mini로 프롬프트 정제
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",   # 최신 ChatGPT 경량 모델
        messages=[
            {"role": "system", "content": "너는 이미지 편집 프롬프트를 명확하게 다듬는 보조자야."},
            {"role": "user", "content": f"다음 프롬프트를 이미지 수정용으로 간결하게 정리해줘:\n{req.prompt}"}
        ]
    )
    refined_prompt = chat_response.choices[0].message.content.strip()

    # 2) 이미지와 마스크 준비
    img_file, mask_file = url_to_png(req.image_url)
    if not img_file:
        return {"error": "이미지 변환 실패"}

    # 3) OpenAI 이미지 수정 API 호출
    files = {
        "image": ("input.png", img_file, "image/png"),
        "mask": ("mask.png", mask_file, "image/png"),
    }
    data = {
        "model": "gpt-image-1",  # DALL·E3 이미지 편집 모델
        "prompt": refined_prompt,
        "n": 1,
        "size": "1024x1024",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers=headers,
        files=files,
        data=data,
    )

    if resp.status_code != 200:
        return {"error": resp.text}

    result = resp.json()
    data_item = result["data"][0]

    # URL 또는 base64 반환
    if "url" in data_item:
        return {"edited_image_url": data_item["url"], "used_prompt": refined_prompt}
    elif "b64_json" in data_item:
        return {"edited_image_base64": data_item["b64_json"], "used_prompt": refined_prompt}
    else:
        return {"error": "이미지 응답 없음"}


# 라우터 앱에 등록
app.include_router(router)
