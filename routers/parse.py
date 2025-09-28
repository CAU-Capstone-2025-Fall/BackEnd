import os
from typing import Optional
from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatParseRequest(BaseModel):
    user_answer: str
    image: Optional[str] = None  # 필요시 이미지도 지원

async def parse_pet_related_info(user_answer: str, image: str | None = None) -> str:
    system_prompt = (
        "아래 사용자의 답변에서 '반려동물 양육에 영향을 주는 요소'를 추출해줘. "
        "추출할 항목: 1) 사는 지역, 2) 집의 형태, 3) 주거지 주변 시설, 4) 집의 크기, 5) 직업, 6) 소득 수준, 7) 가족 구성원 수, 8) 반려동물 양육 경험, 9) 가족의 반려동물 선호도, 10) 가족 구성원의 건강 상태, 11) 여행 빈도, 12) 기타 반려동물 관련 정보. "
        "각 항목을 가능한 한 정확하게 파악해서 JSON 형식으로 응답해줘. "
        "만약 항목에 대한 정보가 없으면 빈 문자열로 표시해줘. "
        """예시: {
            "area": "서울 송파구",
            "house_type": "아파트",
            "nearby_facilities": "공원, 동물병원, 애견카페",
            "house_size": "34평",
            "job": "회사원",
            "income_level": "연소득 6000만원",
            "family_count": 4,
            "pet_experience": "과거에 강아지를 8년간 키운 경험 있음",
            "family_pet_preference": "가족 모두 반려동물에 호의적",
            "family_health_status": "아내는 동물 알레르기 없음, 아들은 천식이 있으나 관리 중",
            "travel_frequency": "연 2회 해외 여행, 월 1회 국내 여행",
            "other_info": "현재는 반려동물 없음, 집 주변에 산책로가 많고, 가족 모두 동물을 좋아함"
            }"""
    )
    user_content = [
        {"type": "input_text", "text": system_prompt + "\n\n" + user_answer}
    ]
    if image:
        user_content.append({"type": "input_image", "image_url": image})

    resp = await client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": user_content}],
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return resp.output_text

@router.post("/parse")
async def chat_parse_pet_info(req: ChatParseRequest):
    info = await parse_pet_related_info(req.user_answer, req.image)
    return {"parsed_info": info}