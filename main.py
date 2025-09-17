import os

from db import crud_api
from db.auto_fetch import start_scheduler
from dotenv import load_dotenv
from fastapi import FastAPI
from routers import chat, items

load_dotenv()  # .env 파일 로드

API_KEY = os.getenv("API_KEY") #추후 데이터 로드할때 필요한 키
app = FastAPI()

app.include_router(items.router)
app.include_router(crud_api.router)
app.include_router(chat.router)
start_scheduler()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Github Actions CI/CD is working!"}