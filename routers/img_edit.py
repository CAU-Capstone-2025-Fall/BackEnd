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

app = FastAPI(title="Image Edit Backend with Auto Mask")
router = APIRouter(prefix="/image", tags=["image"])

class ImageEditRequest(BaseModel):
    prompt: str
    image_url: str


def url_to_png(image_url: str):
    resp = requests.get(image_url)
    if resp.status_code != 200:
        return None
    image = Image.open(BytesIO(resp.content)).convert("RGBA")
    image = image.resize((1024, 1024), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_mask_with_gpt(image_url: str):
    """
    GPT Vision 모델에게 이미지를 보여주고
    '더러운 부분만 흰색, 나머지는 검은색' 마스크 이미지를 생성하게 한다.
    """
    response = client.images.generate(
        model="gpt-image-1",  # 여기서도 이미지 생성 모델 사용
        prompt="Generate a binary mask highlighting only dirty areas (mud, stains, foreign substances) in pure white, keep everything else black. Keep exact same resolution.",
        size="1024x1024",
        image=[{"url": image_url}]
    )
    # 결과는 base64로 반환됨
    b64_img = response.data[0].b64_json
    mask_bytes = base64.b64decode(b64_img)
    return BytesIO(mask_bytes)


@router.post("/edit")
async def edit_image(req: ImageEditRequest):
    # 1. 프롬프트 정제
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 이미지 편집 프롬프트를 DALL·E 편집용으로 정제하는 보조자야."},
            {"role": "user", "content": f"다음 프롬프트를 이미지 수정용으로 간결하게 정리해줘:\n{req.prompt}"}
        ]
    )
    refined_prompt = chat_response.choices[0].message.content.strip()

    # 2. 원본 이미지와 자동 생성된 마스크 준비
    img_file = url_to_png(req.image_url)
    if not img_file:
        return {"error": "이미지 다운로드 실패"}
    mask_file = generate_mask_with_gpt(req.image_url)

    # 3. DALL·E 편집 API 호출
    files = {
        "image": ("input.png", img_file, "image/png"),
        "mask": ("mask.png", mask_file, "image/png"),
    }
    data = {
        "model": "gpt-image-1",
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

    if "url" in data_item:
        return {"edited_image_url": data_item["url"], "used_prompt": refined_prompt}
    elif "b64_json" in data_item:
        return {"edited_image_base64": data_item["b64_json"], "used_prompt": refined_prompt}
    else:
        return {"error": "이미지 응답 없음"}


app.include_router(router)
