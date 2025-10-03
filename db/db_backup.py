from datetime import date, timedelta
import os
from time import sleep
from dotenv import load_dotenv
from fastapi import FastAPI
from pymongo import MongoClient
import requests

app = FastAPI()

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

collection.aggregate([
  { "$match": {} },      # 모든 문서 선택
  { "$out": "new_collection" }
])