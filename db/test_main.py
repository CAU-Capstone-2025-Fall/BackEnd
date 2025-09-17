from fastapi import FastAPI
from crud_api import router
from auto_fetch import start_scheduler

app = FastAPI()

# CRUD API 등록
app.include_router(router)

# APScheduler 시작
start_scheduler()

@app.get("/")
def root():
    return {"msg": "서버 및 자동 데이터 갱신 동작 중"}