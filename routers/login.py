import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

USERS_FILE = Path("users.txt")
SESSIONS = {}

def load_users():
    users = {}
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            for line in f:
                if ":" in line:
                    u, p = line.strip().split(":", 1)
                    users[u] = p
    return users

# 요청 스키마 정의
class UserRequest(BaseModel):
    username: str
    password: str

@router.post("/signup")
def signup(data: UserRequest):
    users = load_users()
    if data.username in users:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")
    with open(USERS_FILE, "a") as f:
        f.write(f"{data.username}:{data.password}\n")
    return {"message": "회원가입 성공"}

@router.post("/login")
def login(data: UserRequest, response: Response):
    users = load_users()
    if data.username not in users or users[data.username] != data.password:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호 오류")
    token = secrets.token_hex(16)
    SESSIONS[token] = data.username
    response.set_cookie("session", token, httponly=True)
    return {"message": "로그인 성공", "user": data.username}

@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token and token in SESSIONS:
        del SESSIONS[token]
    response.delete_cookie("session")
    return {"message": "로그아웃 성공"}

@router.get("/protected")
def protected(request: Request):
    token = request.cookies.get("session")
    if not token or token not in SESSIONS:
        raise HTTPException(status_code=401, detail="로그인 필요")
    return {"message": f"{SESSIONS[token]} 님 환영합니다!"}

def get_current_username(request: Request) -> str:
    sid = request.cookies.get("session")
    if not sid or sid not in SESSIONS:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return SESSIONS[sid]