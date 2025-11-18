import json
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("REPORT_DB", "pet_rec")
COLL_NAME = os.getenv("REPORT_COLL", "reports")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLL_NAME]

print("=== 🔍 Reports 컬렉션 전체 구조 확인 ===")

cursor = collection.find({}, {"_id": 0}).limit(5)
docs = list(cursor)

if not docs:
    print("⚠ 문서가 하나도 없습니다.")
else:
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(json.dumps(doc, indent=4, ensure_ascii=False))
