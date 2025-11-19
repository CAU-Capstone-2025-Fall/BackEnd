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
