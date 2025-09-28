from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import requests
from time import sleep
from datetime import date, timedelta

app = FastAPI()
load_dotenv()

# DB 연결
try:
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)
    db = client["testdb"]
    collection = db["abandoned_animals"]
    collection.create_index("desertionNo", unique=True)
    print ("db connected")
except Exception as e:
    print("db connection error:", e)

def fetch_and_update():
    print("[자동 데이터 갱신] 시작")
    page = 1
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    while True:
        service_key = os.getenv("SERVICE_KEY")
        url = os.getenv("DATAPORTAL_URL")
        params = {
            "serviceKey": service_key,
            "bgnde": yesterday,
            "state": "protect",
            "numOfRows": 50,
            "_type": "json",
            "pageNo": page
        }
        response = requests.get(url, params=params, timeout=10)
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
            collection.update_one(
                {"desertionNo": item["desertionNo"]},
                {"$set": {**item, "createdImg": None}},
                upsert=True
            )
        page += 1
        sleep(0.1)
    print("[자동 데이터 갱신] 완료")

# APScheduler로 매일 새벽 3시 작업 예약
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_and_update, 'cron', hour=3, minute=0)
    scheduler.start()

if __name__ == "__main__":
    fetch_and_update()