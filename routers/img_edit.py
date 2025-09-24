from PIL import Image
import requests
from fastapi import APIRouter
from pydantic import BaseModel
from io import BytesIO
import os

api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter(prefix="/image", tags=["image"])

class ImageEditRequest(BaseModel):
    prompt: str
    image_url: str

def url_to_png(image_url):
    # 이미지 다운로드
    img_resp = requests.get(image_url)
    if img_resp.status_code != 200:
        return {"error": "이미지 다운로드 실패"}
    
    # 이미지 바이트로 오픈
    image = Image.open(BytesIO(img_resp.content))
    image = image.resize((1024, 1024), Image.LANCZOS)
    mask = Image.new("RGBA", image.size, (255, 255, 255, 255))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # PNG로 변환 
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    buf2 = BytesIO()
    mask.save(buf2, format="PNG")
    buf2.seek(0)
    return buf.read(), buf2.read()

@router.post("/edit")
async def edit_image(req: ImageEditRequest):
    img_file, mask_file = url_to_png(req.image_url)  # 이미지 PNG로 변환

    # 2. 이미지 파일로 OpenAI에 업로드
    files = {
        "image": ("input.png", img_file, "image/png"),
        "mask": ("mask.png", mask_file, "image/png")
    }
    data = {
        "prompt": req.prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "url"
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    print("API 요청 프롬프트:", req.prompt)
    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers=headers,
        files=files,
        data=data
    )
    print("응답 코드:", response.status_code)
    print("응답 텍스트:", response.text)

    if response.status_code != 200:
        return {"error": response.text}
    result = response.json()
    image_url = result["data"][0]["url"]
    return {"edited_image_url": image_url}