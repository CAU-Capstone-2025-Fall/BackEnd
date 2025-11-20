# routers/reviews_crud.py
import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pymongo import MongoClient
from routers.login import get_current_username  # ← 로그인 의존성

load_dotenv()

ADMIN_USERS = os.getenv("ADMIN_USERS", "")
ADMIN_USERS = [u.strip() for u in ADMIN_USERS.split(",") if u.strip()]
router = APIRouter(tags=["review"])

# MongoDB 연결
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["testdb"]               # 실제 DB 이름으로 바꿔도 됨
reviews_col = db["reviews"]         # 자동 생성됨 (없으면)

# ----------------------- 유틸 -----------------------
def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _public_url(local_path: str) -> str:
    # uploads/xxx.jpg → /static/xxx.jpg (main.py에서 /static mount 필요)
    return "/static/" + os.path.basename(local_path)

# ----------------------- CRUD -----------------------

@router.get("/reviews")
def list_reviews(skip: int = 0, limit: int = 10):
    """리뷰 목록 조회"""
    cursor = reviews_col.find().sort("createdAt", -1).skip(skip).limit(limit)
    items = [dict(item) for item in cursor]
    total = reviews_col.count_documents({})
    for i in items:
        i["_id"] = str(i["_id"])  # ObjectId → str 변환
    return {"items": items, "total": total}


@router.get("/reviews/{rid}")
def get_review(rid: str):
    """특정 리뷰 조회"""
    doc = reviews_col.find_one({"_id": ObjectId(rid)}) if ObjectId.is_valid(rid) else reviews_col.find_one({"_id": rid})
    if not doc:
        raise HTTPException(404, "존재하지 않는 글")
    doc["_id"] = str(doc["_id"])
    return doc


@router.post("/reviews")
def create_review(
    username: str = Depends(get_current_username),
    title: str = Form(...),
    body: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """리뷰 작성"""
    image_urls = []
    if files:
        os.makedirs("uploads", exist_ok=True)
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            name = f"{int(datetime.utcnow().timestamp())}_{ObjectId()}{ext or '.bin'}"
            path = os.path.join("uploads", name)
            with open(path, "wb") as w:
                w.write(f.file.read())
            image_urls.append(_public_url(path))

    now = _now_iso()
    doc = {
        "title": title,
        "body": body,
        "images": image_urls,
        "authorId": username,
        "authorName": username,
        "createdAt": now,
        "updatedAt": now,
    }

    result = reviews_col.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


@router.patch("/reviews/{rid}")
def update_review(
    rid: str,
    username: str = Depends(get_current_username),
    title: Optional[str] = Form(None),
    body: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """리뷰 수정"""
    doc = reviews_col.find_one({"_id": ObjectId(rid)}) if ObjectId.is_valid(rid) else reviews_col.find_one({"_id": rid})
    if not doc:
        raise HTTPException(404, "존재하지 않는 글")

    if username not in ADMIN_USERS and oc.get("authorId") != username:
        raise HTTPException(403, "작성자만 수정 가능")

    update_fields = {}
    if title is not None:
        update_fields["title"] = title
    if body is not None:
        update_fields["body"] = body

    if files:
        os.makedirs("uploads", exist_ok=True)
        new_urls = []
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            name = f"{int(datetime.utcnow().timestamp())}_{ObjectId()}{ext or '.bin'}"
            path = os.path.join("uploads", name)
            with open(path, "wb") as w:
                w.write(f.file.read())
            new_urls.append(_public_url(path))
        update_fields["images"] = new_urls

    update_fields["updatedAt"] = _now_iso()

    reviews_col.update_one({"_id": doc["_id"]}, {"$set": update_fields})
    updated = reviews_col.find_one({"_id": doc["_id"]})
    updated["_id"] = str(updated["_id"])
    return updated


@router.delete("/reviews/{rid}")
def delete_review(rid: str, username: str = Depends(get_current_username)):
    """리뷰 삭제"""
    doc = reviews_col.find_one({"_id": ObjectId(rid)}) if ObjectId.is_valid(rid) else reviews_col.find_one({"_id": rid})
    if not doc:
        raise HTTPException(404, "존재하지 않는 글")

    if username not in ADMIN_USERS and doc.get("authorId") != username:
        raise HTTPException(403, "작성자만 삭제 가능")

    reviews_col.delete_one({"_id": doc["_id"]})
    return {"ok": True}
