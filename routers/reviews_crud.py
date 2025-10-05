# routers/reviews_crud.py
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
from threading import RLock

from bson import ObjectId  
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Request
from routers.login import protected

router = APIRouter()

# ------------------ 파일 저장 설정 ------------------
REVIEWS_FILE = Path("reviews.json")   # 서버에 하나만 존재하는 리뷰 저장소(공유)
FILE_LOCK = RLock()

def _now_iso() -> str:
    # ISO 8601 (UTC)
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_file():
    """reviews.json이 없으면 빈 배열로 초기화"""
    if not REVIEWS_FILE.exists():
        REVIEWS_FILE.write_text("[]", encoding="utf-8")

def _read_all() -> List[dict]:
    _ensure_file()
    with FILE_LOCK:
        data = json.loads(REVIEWS_FILE.read_text(encoding="utf-8") or "[]")
        if not isinstance(data, list):
            # 혹시 파일이 손상된 경우 복구
            return []
        return data

def _atomic_write(all_items: List[dict]):
    with FILE_LOCK:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="reviews_", suffix=".json")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, REVIEWS_FILE)  
        except Exception:
            # 실패 시 임시파일 제거
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

# ------------------ 유틸 ------------------
def _public_url(local_path: str) -> str:
    # "uploads/xxx.jpg" -> "/static/xxx.jpg" (main.py에서 /static 마운트 필요)
    return "/static/" + os.path.basename(local_path)

def _username_from_protected(request: Request) -> str:
    msg = protected(request)["message"]
    suffix = " 님 환영합니다!"
    return msg[:-len(suffix)] if msg.endswith(suffix) else msg

# =========================================================
#                         CRUD
# =========================================================

@router.get("/reviews")
def list_reviews(skip: int = 0, limit: int = 10):

    items = _read_all()
    # 정렬: createdAt 내림차순
    items_sorted = sorted(items, key=lambda x: x.get("createdAt", ""), reverse=True)
    total = len(items_sorted)
    return {
        "items": items_sorted[skip:skip + limit],
        "total": total
    }

@router.get("/reviews/{rid}")
def get_review(rid: str):
    items = _read_all()
    for doc in items:
        if doc.get("_id") == rid:
            return doc
    raise HTTPException(404, "존재하지 않는 글")

@router.post("/reviews")
def create_review(
    request: Request,
    title: str = Form(...),
    body: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):

    username = _username_from_protected(request)

    # 이미지 저장 (개발: 로컬 uploads/)
    image_urls = []
    if files:
        os.makedirs("uploads", exist_ok=True)
        for f in files:
            ext = os.path.splitext(f.filename)[1].lower()
            # 파일명: 타임스탬프 + ObjectId + 확장자
            name = f"{int(datetime.utcnow().timestamp())}_{ObjectId()}{ext}"
            path = os.path.join("uploads", name)
            with open(path, "wb") as w:
                w.write(f.file.read())
            image_urls.append(_public_url(path))

    now = _now_iso()
    doc = {
        "_id": str(uuid4()),   # 파일 기반이라 uuid로 고유 id 생성
        "title": title,
        "body": body,
        "images": image_urls,
        "authorId": username,
        "authorName": username,
        "createdAt": now,
        "updatedAt": now
    }

    items = _read_all()
    items.append(doc)
    _atomic_write(items)
    return doc

@router.patch("/reviews/{rid}")
def update_review(
    request: Request,
    rid: str,
    title: Optional[str] = Form(None),
    body: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    후기 수정 (작성자만, 부분 수정)
    파일을 보내면 기존 이미지 전체 교체 정책
    """
    username = _username_from_protected(request)
    items = _read_all()

    # 대상 찾기
    idx = next((i for i, d in enumerate(items) if d.get("_id") == rid), None)
    if idx is None:
        raise HTTPException(404, "존재하지 않는 글")

    doc = items[idx]
    if doc.get("authorId") != username:
        raise HTTPException(403, "작성자만 수정 가능")

    # 필드 갱신
    if title is not None:
        doc["title"] = title
    if body is not None:
        doc["body"] = body

    # 이미지 교체
    if files:
        os.makedirs("uploads", exist_ok=True)
        new_urls = []
        for f in files:
            ext = os.path.splitext(f.filename)[1].lower()
            name = f"{int(datetime.utcnow().timestamp())}_{ObjectId()}{ext}"
            path = os.path.join("uploads", name)
            with open(path, "wb") as w:
                w.write(f.file.read())
            new_urls.append(_public_url(path))
        doc["images"] = new_urls

    doc["updatedAt"] = _now_iso()
    items[idx] = doc
    _atomic_write(items)
    return doc

@router.delete("/reviews/{rid}")
def delete_review(request: Request, rid: str):
    """
    후기 삭제 (작성자만)
    """
    username = _username_from_protected(request)
    items = _read_all()

    idx = next((i for i, d in enumerate(items) if d.get("_id") == rid), None)
    if idx is None:
        raise HTTPException(404, "존재하지 않는 글")

    doc = items[idx]
    if doc.get("authorId") != username:
        raise HTTPException(403, "작성자만 삭제 가능")

    # 삭제 후 저장
    del items[idx]
    _atomic_write(items)
    return {"ok": True}
