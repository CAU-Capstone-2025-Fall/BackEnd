import os
from typing import Optional

from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

# OpenAI 클라이언트 초기화 (환경변수에서 API 키 읽기)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 라우터 설정
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None  # URL 이미지 지원 가능

# OpenAI 호출 함수
async def ask_chatbot(message: str, image: str | None = None) -> str:
    user_content = [{"type": "input_text", "text": message}]
    if image:
        user_content.append({"type": "input_image", "image_url": image})

    resp = await client.responses.create(
        model="gpt-5-nano",  # 멀티모달 지원 모델
        input=[{"role": "user", "content": user_content}],
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )

    return resp.output_text

@router.post("/ask")
async def chat_with_bot(req: ChatRequest):
    answer = await ask_chatbot(req.message, req.image)
    return {"answer": answer}
