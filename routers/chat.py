import os

from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

# OpenAI 클라이언트 초기화
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 라우터 설정
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str  # 그냥 텍스트만 받음

# OpenAI 호출 함수 (텍스트 전용 + 역할 부여)
async def ask_chatbot(message: str) -> str:
    resp = await client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "너는 친절한 상담가야. 유기 강아지를 입양하고 싶은 사용자의 고민을 잘 들어주고 따뜻하게 답해줘."},
            {"role": "user", "content": message},
        ],
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return resp.output_text

@router.post("/ask")
async def chat_with_bot(req: ChatRequest):
    answer = await ask_chatbot(req.message)
    return {"answer": answer}
