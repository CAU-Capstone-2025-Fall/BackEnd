# from fastapi import FastAPI
# from apscheduler.schedulers.background import BackgroundScheduler
# import os
# from dotenv import load_dotenv
# from pymongo import MongoClient
# import requests
# from time import sleep
# from datetime import date, timedelta

# app = FastAPI()
# load_dotenv()

# # DB 연결
# try:
#     uri = os.getenv("MONGODB_URI")
#     client = MongoClient(uri)
#     db = client["testdb"]
#     collection = db["abandoned_animals"]
#     collection.create_index("desertionNo", unique=True)
#     print ("db connected")
# except Exception as e:
#     print("db connection error:", e)

# lista = ["431370202500931",
#     "411302202500843",
#     "441560202501760",
#     "426333202501140",
#     "441409202501712"]
# i = 0

# def fetch_and_update():
#     print("[자동 데이터 갱신] 시작")
#     page = 1
#     yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
#     while True:
#         service_key = os.getenv("SERVICE_KEY")
#         url = os.getenv("DATAPORTAL_URL")
#         params = {
#             "serviceKey": service_key,
#             # "desertion_no": "441409202501712",
#             "bgnde": 20251121,
#             "state": "protect",
#             "numOfRows": 50,
#             "_type": "json",
#             "pageNo": page
#         }
#         response = requests.get(url, params=params)
#         try:
#             data = response.json()
#             items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
#         except Exception as e:
#             print("JSON 파싱 실패. 원본 응답:", response.text)
#             data = None
#             items = []
#         if not items:
#             break
#         for item in items:
#             print("처리 중인 유기동물 번호:", item.get("desertionNo"))
#             collection.update_one(
#                 {"desertionNo": item["desertionNo"]},
#                 {"$set": {**item, "createdImg": None, "extractedFeature": None, "improved":"0"}},
#                 upsert=True
#             )
#         break
#         page += 1
#         sleep(0.1)
#     print("[자동 데이터 갱신] 완료")

# def fetch_improved():
#     result = collection.update_many(
#         {"improve": {"$exists": False}},
#         {"$set": {"improve": "0"}}
#     )
#     print("업데이트된 문서 수:", result.modified_count)

# # APScheduler로 매일 새벽 3시 작업 예약
# def start_scheduler():
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(fetch_and_update, 'cron', hour=3, minute=0)
#     scheduler.start()

# if __name__ == "__main__":
#     fetch_and_update()
#     # fetch_improved()