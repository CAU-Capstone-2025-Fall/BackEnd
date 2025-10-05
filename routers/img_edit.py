import base64
import os
from io import BytesIO

import requests
from fastapi import APIRouter, FastAPI
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI(title="Chat-like Image Editor")
router = APIRouter(prefix="/image", tags=["image"])

class ImageEditRequest(BaseModel):
    prompt: str
    image_url: str


def url_to_png(image_url: str):
    """이미지를 URL에서 받아 PNG로 변환"""
    resp = requests.get(image_url)
    if resp.status_code != 200:
        return None
    image = Image.open(BytesIO(resp.content)).convert("RGBA")
    image = image.resize((1024, 1024), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


@router.post("/chat-edit")
async def chat_edit(req: ImageEditRequest):
    """
    채팅창처럼 → GPT가 편집 지시(JSON) 반환 → 백엔드가 실행 → 최종 이미지 반환
    """
    # 1. GPT에게 이미지 편집 API 호출 스펙(JSON) 생성 요청
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 OpenAI Images API 전문가다. 사용자의 요청을 기반으로 이미지 편집 API 호출 사양(JSON)을 만들어라."},
            {"role": "user", "content": f"원본 이미지 URL: {req.image_url}\n프롬프트: {req.prompt}\n\nOpenAI `images/edits` API에 넣을 JSON 스펙을 만들어줘. keys: model, prompt, n, size"}
        ],
        response_format={ "type": "json_object" }
    )

    spec = chat_response.choices[0].message.content
    print("GPT가 만든 API 스펙:", spec)

    # 2. 원본 이미지 준비
    img_file = url_to_png(req.image_url)
    if not img_file:
        return {"error": "이미지 다운로드 실패"}

    # 3. 실제 이미지 편집 API 호출
    files = {
        "image": ("input.png", img_file, "image/png"),
    }
    data = {
        "model": "gpt-image-1",  # GPT가 제안한 모델 고정
        "prompt": req.prompt,    # GPT가 정제한 prompt 대신 원본 prompt 사용 가능
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

    # 4. 최종 결과 반환
    if "url" in data_item:
        return {"edited_image_url": data_item["url"], "used_spec": spec}
    elif "b64_json" in data_item:
        return {"edited_image_base64": data_item["b64_json"], "used_spec": spec}
    else:
        return {"error": "이미지 응답 없음"}


app.include_router(router)
