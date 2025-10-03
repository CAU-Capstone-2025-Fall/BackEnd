import os

from db import crud_api
from db.auto_fetch import start_scheduler
from dotenv import load_dotenv
from fastapi import FastAPI
from routers import chat, img_edit, items, login, parse

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()
app.include_router(items.router)
app.include_router(crud_api.router)
app.include_router(chat.router)
app.include_router(img_edit.router)
app.include_router(parse.router)
app.include_router(login.router)   # 로그인 라우터 연결

#@app.on_event("startup")
#def startup_event():
#    start_scheduler()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Github Actions CI/CD is working!"}
