import json
import os

import numpy as np
from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 초기 설정
# ------------------------------------------------------------
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
router = APIRouter(prefix="/chat", tags=["chat"])

# 미리 임베딩된 RAG 데이터 로드
DATA_PATH = "data/multi_embedded.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)
print(f"DATA 총 개수: {len(DATA)}")
for i, item in enumerate(DATA[:3]):  # 앞의 3개만 확인
    print(f"[{i}] answer: {item.get('answer')[:30]}")
    print(f"   queries 개수: {len(item.get('queries', []))}")
    print(f"   embedded_queries 개수: {len(item.get('embedded_queries', []))}")
    print(f"   embedded_queries[0] 길이: {len(item['embedded_queries'][0]) if item['embedded_queries'] else 0}")

# ------------------------------------------------------------
# Pydantic 모델
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
async def get_embedding(text: str):
    """입력 문장을 OpenAI embedding vector로 변환"""
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding)


def compute_similarity(user_emb, group_embs):
    """사용자 임베딩과 그룹 내 쿼리 임베딩들의 최대 유사도 반환"""
    sims = cosine_similarity([user_emb], np.array(group_embs))[0]
    return np.max(sims)


async def retrieve_best_answer(user_query: str, threshold: float = 0.75):
    """RAG 검색: 가장 유사한 답변을 찾되, threshold 이하이면 None 반환"""
    user_emb = await get_embedding(user_query)

    best_score = -1
    best_answer = None

    for item in DATA:
        score = compute_similarity(user_emb, item["embedded_queries"])
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score < threshold:
        return None, best_score  # 관련 문서 없음
    return best_answer, best_score


# ------------------------------------------------------------
# 챗봇 응답 함수 (RAG + fallback)
# ------------------------------------------------------------
async def ask_chatbot(message: str) -> str:
    # 1️⃣ RAG 검색 시도
    best_answer, score = await retrieve_best_answer(message)

    # 2️⃣ 유사한 문서가 충분하면 RAG 기반 응답
    if best_answer:
        system_prompt = (
            "너는 유기동물 입양 전문 상담가야. "
            "아래 참고 문서를 바탕으로 사용자의 질문에 따뜻하고 정확하게 답변해줘. 너무 길게 이야기하진 말아줘."
        )
        user_content = f"참고 문서:\n{best_answer}\n\n질문:\n{message}"

    # 3️⃣ 유사도가 낮으면 (관련 없음) → 기존 OpenAI 기본 모드로 대체
    else:
        system_prompt = (
            "너는 친절한 상담가야. "
            "유기 강아지를 입양하고 싶은 사용자의 고민을 들어주고 따뜻하게 답해줘. 너무 길게 이야기하진 말아줘."
        )
        user_content = message

    # 4️⃣ 실제 OpenAI 호출
    resp = await client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return resp.output_text


# ------------------------------------------------------------
# FastAPI 엔드포인트
# ------------------------------------------------------------
@router.post("/ask")
async def chat_with_bot(req: ChatRequest):
    answer = await ask_chatbot(req.message)
    return {"answer": answer}