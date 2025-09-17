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
    params = {
        "serviceKey": service_key,
        "state": "protect",
        "numOfRows": 500,
        "_type" : "json",
        "pageNo" : page
    }

    response = requests.get(
        url, 
        params=params,
    )
    data = response.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) 

    if not items:
        break
    
    for item in items:
        collection.update_one(
        {"desertionNo": item["desertionNo"]},
        {"$set": item},
        upsert=True
        )
    print(f"Inserted page: {page}")
    page += 1
    sleep(0.01)

print("Data fetching and insertion completed.")
