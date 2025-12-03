import os
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from filed_embedding import doc_to_4texts, batch_embed_texts

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
EMBED_MODEL = "text-embedding-3-small"

client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]

# def get_embedding(text: str):
#     response = openai.embeddings.create(
#         model=EMBED_MODEL,
#         input=text
#     )
#     return response.data[0].embedding

# for doc in collection.find():
#     # 1. 추출값(특징) 모으기
#     feat = doc.get("extractedFeature", "")
#     # dict이면 value만 빈값빼고 합치기 (문장화)
#     if isinstance(feat, dict):
#         text = " ".join([str(v) for v in feat.values() if v])
#     else:
#         text = str(feat)
#     # 이미 embedding 있다면 skip (원하면 제외 가능)
#     if doc.get("embedding"):
#         continue
#     try:
#         emb = get_embedding(text)
#         # 2. DB에 바로 embedding 필드 업데이트
#         collection.update_one(
#             {'_id': doc['_id']},
#             {'$set': {'embedding': emb}}
#         )
#         print(f"{doc.get('desertionNo')} 임베딩 저장 완료")
#     except Exception as e:
#         print(f"{doc.get('desertionNo')} 오류: {e}")


# 색깔, 활동성, 외모, 성격 필드 텍스트로 변환
def get_embeddings_batch(texts):
    resp = client_ai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(r.embedding, dtype=np.float32) for r in resp.data]

def persist(doc_id, emb_map):
    # emb_map: dict field->list floats (serializable)
    collection.update_one({"desertionNo": doc_id}, {"$set": {"fieldEmbeddings": emb_map}}, upsert=False)

def main(batch_size=200):
    # cursor = collection.find({}, {"desertionNo":1, "kindFullNm":1, "colorCd":1, "specialMark":1, "extractedFeature":1})
    cursor = collection.find({}, {"desertionNo":1, "kindFullNm":1, "colorCd":1, "specialMark":1, "extractedFeature":1})
    batch = []
    ids = []
    for doc in cursor:
        fields = doc_to_4texts(doc)
        texts = [v for v in fields.values() if v]
        keys = [k for k,v in fields.items() if v]
        if not texts:
            continue
        # embed in batches per document to avoid mixing documents; for performance you could aggregate across docs
        vecs = batch_embed_texts(get_embeddings_batch, texts)
        emb_map = {k: v.tolist() for k,v in zip(keys, vecs)}
        collection.update_one({"desertionNo": str(doc.get("desertionNo"))}, {"$set": {"fieldEmbeddings": emb_map}})
        print(f"Doc {doc.get('desertionNo')} field embeddings saved.")
    print("done")

if __name__ == "__main__":
    main()