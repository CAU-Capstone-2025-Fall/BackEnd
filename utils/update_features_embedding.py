import os
from pymongo import MongoClient
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
EMBED_MODEL = "text-embedding-3-small"

openai.api_key = OPENAI_API_KEY
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]

def get_embedding(text: str):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

for doc in collection.find():
    # 1. 추출값(특징) 모으기
    feat = doc.get("extractedFeature", "")
    # dict이면 value만 빈값빼고 합치기 (문장화)
    if isinstance(feat, dict):
        text = " ".join([str(v) for v in feat.values() if v])
    else:
        text = str(feat)
    # 이미 embedding 있다면 skip (원하면 제외 가능)
    if doc.get("embedding"):
        continue
    try:
        emb = get_embedding(text)
        # 2. DB에 바로 embedding 필드 업데이트
        collection.update_one(
            {'_id': doc['_id']},
            {'$set': {'embedding': emb}}
        )
        print(f"{doc.get('desertionNo')} 임베딩 저장 완료")
    except Exception as e:
        print(f"{doc.get('desertionNo')} 오류: {e}")