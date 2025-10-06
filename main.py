import os

from db import crud_api

#from db.auto_fetch import start_scheduler
from dotenv import load_dotenv
from fastapi import FastAPI
from routers import chat, img_edit, items, login, parse
from routers import reviews_crud , favorite
from fastapi.staticfiles import StaticFiles

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="uploads"), name="static")
app.include_router(items.router)
app.include_router(crud_api.router)
app.include_router(chat.router)
app.include_router(img_edit.router)
app.include_router(parse.router)
app.include_router(login.router)   # 로그인 라우터 연결
app.include_router(reviews_crud.router)
app.include_router(favorite.router)  

#@app.on_event("startup")
#def startup_event():
#    start_scheduler()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Github Actions CI/CD is working!"}
