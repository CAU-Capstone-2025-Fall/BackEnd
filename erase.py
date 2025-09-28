import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()  # .env 파일 로드

try:
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)
    db = client["testdb"]
    collection = db["abandoned_animals"]
    collection.create_index("desertionNo", unique=True)
    print ("db connected")
except Exception as e:
    print("db connection error:", e)

collection.delete_many({})  # 모든 문서 삭제