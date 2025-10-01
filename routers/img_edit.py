import os
from io import BytesIO

import requests
from fastapi import APIRouter
from PIL import Image
from pydantic import BaseModel

api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter(prefix="/image", tags=["image"])

class ImageEditRequest(BaseModel):
    prompt: str
    image_url: str

def url_to_png(image_url):
    # 이미지 다운로드
    img_resp = requests.get(image_url)
    if img_resp.status_code != 200:
        return None, None
    
    # 원본 이미지 RGBA 변환
    image = Image.open(BytesIO(img_resp.content)).convert("RGBA")
    image = image.resize((1024, 1024), Image.LANCZOS)

    # PNG 버퍼 생성
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # 마스크: 전체 투명 (→ 전체 영역 수정 가능)
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    buf_mask = BytesIO()
    mask.save(buf_mask, format="PNG")
    buf_mask.seek(0)

    return buf, buf_mask
@router.post("/edit")
async def edit_image(req: ImageEditRequest):
    img_file, mask_file = url_to_png(req.image_url)
    if not img_file:
        return {"error": "이미지 변환 실패"}

    files = {
        "image": ("input.png", img_file, "image/png"),
        "mask": ("mask.png", mask_file, "image/png"),
    }
    data = {
        "model": "gpt-image-1",   # DALL·E 3 최신 모델
        "prompt": req.prompt,
        "n": 1,
        "size": "1024x1024",
        # "response_format": "url",  # url 모드 강제 (선택)
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers=headers,
        files=files,
        data=data,
    )
    print("응답 코드:", response.status_code)
    print("응답 텍스트:", response.text)

    if response.status_code != 200:
        return {"error": response.text}

    result = response.json()
    data_item = result["data"][0]

    if "url" in data_item:
        return {"edited_image_url": data_item["url"]}
    elif "b64_json" in data_item:
        import base64
        image_bytes = base64.b64decode(data_item["b64_json"])
        # 필요하다면 파일로 저장하거나 프론트에 직접 전달
        return {"edited_image_base64": data_item["b64_json"]}
    else:
        return {"error": "응답에서 이미지 URL/Base64를 찾을 수 없음"}
