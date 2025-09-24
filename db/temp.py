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

page = 1

while True:
    service_key = os.getenv("SERVICE_KEY")
    url = os.getenv("DATAPORTAL_URL")
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    params = {
        "serviceKey": service_key,
        "bgnde": "20240101",
        "state": "protect",
        "numOfRows": 100,
        "_type" : "json",
        "pageNo" : page
    }

    response = requests.get(
        url, 
        params=params,
        timeout=10,
    )
    try:
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    except Exception as e:
        print("JSON 파싱 실패. 원본 응답:", response.text)
        data = None
        items = []

    if not items:
        break
    
    for item in items:
        try:
            result = collection.update_one(
            {"desertionNo": item["desertionNo"]},
            {"$set": {**item, "createdImg": None}},
            upsert=True
            )
        except Exception as e:
            print("Error inserting item:", e)
            continue
    print(f"Inserted page: {page}")
    page += 1
    sleep(0.1)

print("Data fetching and insertion completed.")
