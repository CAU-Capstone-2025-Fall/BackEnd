import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, FastAPI, Query, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
import re
import json
import numpy as np

router = APIRouter(prefix="/recommand", tags=["recommand"])

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI    = os.getenv("MONGODB_URI")
MODEL          = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]

app = FastAPI()

def get_embedding(text: str):
    """
    OpenAI API로 텍스트 임베딩 벡터 얻기
    """
    resp = client_ai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def extract_species(natural_query: str):
    species_map = {
        "개": ["개", "강아지", "dog", "멍멍이"],
        "고양이": ["고양이", "냥이", "cat"],
        "기타": ["기타", "토끼", "햄스터", "기니피그", "고슴도치", "거북이", "새"]  # 필요에 따라 확장
    }
    q = natural_query.lower()
    for key, keywords in species_map.items():
        for word in keywords:
            if word in q:
                return key
    return None

class RecommendRequest(BaseModel):
    natural_query: str
    limit: int = 3

@router.post("")
def recommend_animals(body: RecommendRequest):
    # 1. 사용자 쿼리 임베딩
    try:
        user_emb = get_embedding(body.natural_query)
    except Exception as e:
        raise HTTPException(500, f"임베딩 생성 오류: {e}")

    # 2. 모든 동물 문서 순회, (extractedFeature 등으로 임베딩)
    species = extract_species(body.natural_query)
    db_species_field = "upKindNm"
    candidates = []
    query_filter = {}
    if species:
        query_filter = {db_species_field: species}
    for doc in collection.find(query_filter):
        feat = doc.get("extractedFeature", "")
        # 특징이 dict면 주요 value 합쳐서 문장화
        # if isinstance(feat, dict):
        #     feat_str = " ".join([str(v) for v in feat.values() if v])
        # else:
        #     feat_str = str(feat)
        # try:
        #     animal_emb = get_embedding(feat_str)
        animal_emb = np.array(doc.get("embedding", []))
        # except Exception:
        #     continue  # 임베딩 실패시 skip
        score = cosine_similarity(user_emb, animal_emb)
        candidates.append((score, doc))

    topn = sorted(candidates, key=lambda x: x[0], reverse=True)[:body.limit]
    return [
        {
            "score": round(s, 4),
            "desertionNo": doc.get("desertionNo"),
            "kindFullNm": doc.get("kindFullNm"),
            # 필요한 정보 추가
        }
        for (s, doc) in topn if s > 0.001  # 0.1 threshold 예시
    ]

## === 파싱 및 유사도 함수 === ##

# def parse_natural_language_query(nl_query: str) -> dict:
#     """
#     사용자 자연어 입력 → 추천용 표준 JSON 구조로 변환(GPT)
#     """
#     prompt = (
#         "다음 문장이 사용자가 원하는 동물의 외형 및 특징입니다. "
#         "아래 12개 key로 JSON을 만들어줘(없으면 null, 예시는 생략):\n"
#         "upKindNm(반드시 개, 고양이, 기타 중 하나), main_color, sub_color, fur_pattern, fur_length, fur_texture, rough_size, "
#         "eye_shape, ear_shape, tail_shape, noticeable_features, health_impression\n"
#         "자연어: " + nl_query
#     )
#     messages = [{"role": "user", "content": prompt}]
#     response = client_ai.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=0.1,
#         max_tokens=400,
#     )
#     output = response.choices[0].message.content.strip()
#     # JSON만 추출
#     match = re.search(r'\{.*\}', output, re.DOTALL)
#     if match:
#         try:
#             parsed = json.loads(match.group(0))
#             return parsed
#         except Exception:
#             pass
#     try:  # 혹시 전체가 json이면
#         return json.loads(output)
#     except Exception:
#         raise HTTPException(status_code=400, detail="GPT가 추천용 JSON을 반환하지 못했습니다.")
    
# def feature_similarity(user: dict, item: dict) -> float:
#     """
#     user, item: 둘 다 추천용 특징 dict(JSON) 형태
#     - 주요 key가 완전히 일치하면 +1, 부분 일치시 +0.5, null/빈값은 점수 없음 등 단순 로직
#     """
#     score = 0
#     keys = [k for k in user if k != "upKindNm"]
#     for k in keys:
#         user_val = (user[k] or "").strip() if k in user else ""
#         item_val = (item.get(k) or "").strip()
#         # 완전일치
#         if user_val and item_val:
#             if user_val == item_val:
#                 score += 1
#             elif user_val in item_val or item_val in user_val:
#                 score += 0.5
#     return score / max(len(keys), 1)

# def enrich_row(doc: dict) -> dict:
#     """
#     extractedFeature를 dict로, 기타 그대로
#     """
#     feature_raw = doc.get("extractedFeature", {})
#     if isinstance(feature_raw, str):
#         try:
#             feature = json.loads(feature_raw)
#         except Exception:
#             feature = {}
#     else:
#         feature = feature_raw
#     doc["extractedFeature"] = feature
#     return doc

# ## ========== API ========== ##
# class RecommendRequest(BaseModel):
#     natural_query: str
#     limit: Optional[int] = 3

# @app.post("/recommend")
# def recommend_animals(body: RecommendRequest):
#     # 1. 사용자의 요구를 표준 구조로 변환(GPT)
#     try:
#         user_feature = parse_natural_language_query(body.natural_query)
#     except Exception as e:
#         raise HTTPException(400, f"자연어 파싱 실패: {str(e)}")
#     user_upkind = user_feature.get("upKindNm")
#     # 2. 전체 animal doc iterate, 추천점수 산출
#     candidates = []
#     if user_upkind:
#         filtered_collection = collection.find({"upKindNm": user_upkind})
#     else:
#         filtered_collection = collection.find()
#     for doc in filtered_collection:
#         full = enrich_row(doc)
#         db_feature = full.get("extractedFeature", {})
#         if not isinstance(db_feature, dict) or not db_feature:
#             continue
#         sim = feature_similarity(user_feature, db_feature)
#         candidates.append( (sim, full) )
#     # 3. 추천 score 내림차순 정렬 & 제한
#     topn = sorted(candidates, key=lambda x: x[0], reverse=True)[:body.limit]
#     # 4. 기타(다른 정보 포함) 전체 리턴
#     return [
#         {
#             "score": s,
#             "desertionNo": doc.get("desertionNo"),
#             # "kindFullNm": doc.get("kindFullNm"),
#             # "age": doc.get("age"),
#             # "weight": doc.get("weight"),
#             # "popfile1": doc.get("popfile1"),
#             # "specialMark": doc.get("specialMark"),
#             # "careNm": doc.get("careNm"),
#             # "careAddr": doc.get("careAddr"),
#             # "noticeSdt": doc.get("noticeSdt"),
#             # "noticeEdt": doc.get("noticeEdt"),
#             # "processState": doc.get("processState"),
#             # "happenPlace": doc.get("happenPlace"),
#             # "extractedFeature": doc.get("extractedFeature"),
#         }
#         for (s, doc) in topn if s > 0
#     ]

# 실행: uvicorn animal_recommender_api:app --reload