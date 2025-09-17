import os

from db import crud_api
from db.auto_fetch import start_scheduler
from dotenv import load_dotenv
from fastapi import FastAPI
from routers import chat, items

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()
app.include_router(items.router)
app.include_router(crud_api.router)
app.include_router(chat.router)

@app.on_event("startup")
def startup_event():
    start_scheduler()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Github Actions CI/CD is working!"}
