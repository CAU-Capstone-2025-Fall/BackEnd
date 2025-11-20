import os

from db import crud_api
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# ------------------------------
# 라우터 import (routers 패키지 내부)
# ------------------------------
from routers import (
    chat,
    encode,
    favorite,
    gptgenerator,
    img_edit,
    items,
    login,
    parse,
    recommend,
    report,
    reviews_crud,
    user_info,
)

# ------------------------------
# 환경 변수 로드
# ------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ------------------------------
# FastAPI 앱 초기화
# ------------------------------
app = FastAPI()

os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# ------------------------------
# 라우터 등록
# ------------------------------
app.include_router(items.router)
app.include_router(crud_api.router)
app.include_router(chat.router)
app.include_router(img_edit.router)
app.include_router(parse.router)
app.include_router(login.router)
app.include_router(reviews_crud.router)
app.include_router(favorite.router)
app.include_router(recommend.router)
app.include_router(encode.router)   
app.include_router(report.router)  # ← 여기서 encodeA 엔드포인트가 활성화됨
app.include_router(user_info.router)
app.include_router(gptgenerator.router)
# ------------------------------
# 루트 엔드포인트
# ------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Github Actions CI/CD is working!"}
