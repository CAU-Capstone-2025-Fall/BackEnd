# routers/login.py
import secrets
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, Response

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

@router.post("/signup")
def signup(username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")
    with open(USERS_FILE, "a") as f:
        f.write(f"{username}:{password}\n")
    return {"message": "회원가입 성공"}

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...), response: Response = None):
    users = load_users()
    if username not in users or users[username] != password:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호 오류")
    token = secrets.token_hex(16)
    SESSIONS[token] = username
    response.set_cookie("session", token, httponly=True)
    return {"message": "로그인 성공", "user": username}

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
