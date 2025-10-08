import os
import secrets

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])

# MongoDB 연결
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["testdb"]        # 👉 실제 DB 이름으로 바꿔도 됨
users_collection = db["users"]

SESSIONS = {}  # 메모리 세션 (테스트용)

# 요청 스키마
class UserRequest(BaseModel):
    username: str
    password: str


# 회원가입
@router.post("/signup")
def signup(data: UserRequest):
    existing = users_collection.find_one({"username": data.username})
    if existing:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")

    users_collection.insert_one({
        "username": data.username,
        "password": data.password,  # 👉 평문 저장
        "persona": {},              # 나중에 설문 데이터 넣을 공간
    })
    return {"message": "회원가입 성공"}


# 로그인
@router.post("/login")
def login(data: UserRequest, response: Response):
    user = users_collection.find_one({"username": data.username})
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호 오류")

    token = secrets.token_hex(16)
    SESSIONS[token] = data.username
    response.set_cookie("session", token, httponly=True)
    return {"message": "로그인 성공", "user": data.username}


# 로그아웃
@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token and token in SESSIONS:
        del SESSIONS[token]
    response.delete_cookie("session")
    return {"message": "로그아웃 성공"}


# 보호된 라우트
@router.get("/protected")
def protected(request: Request):
    token = request.cookies.get("session")
    if not token or token not in SESSIONS:
        raise HTTPException(status_code=401, detail="로그인 필요")
    username = SESSIONS[token]
    return {"message": f"{username} 님 환영합니다!"}


def get_current_username(request: Request) -> str:
    sid = request.cookies.get("session")
    if not sid or sid not in SESSIONS:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return SESSIONS[sid]
