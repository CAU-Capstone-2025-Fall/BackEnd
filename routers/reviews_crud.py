# routers/reviews_crud.py
import json, os, tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
from threading import RLock

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Depends
from fastapi import Request
from routers.login import get_current_username  # ← 위에서 만든 의존성

router = APIRouter()

REVIEWS_FILE = Path("reviews.json")
FILE_LOCK = RLock()

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_file():
    if not REVIEWS_FILE.exists():
        REVIEWS_FILE.write_text("[]", encoding="utf-8")

def _read_all() -> List[dict]:
    _ensure_file()
    with FILE_LOCK:
        txt = REVIEWS_FILE.read_text(encoding="utf-8") or "[]"
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            # 손상 복구
            REVIEWS_FILE.with_suffix(".bad.json").write_text(txt, encoding="utf-8")
            data = []
        return data if isinstance(data, list) else []

def _atomic_write(all_items: List[dict]):
    with FILE_LOCK:
        fd, tmp_path = tempfile.mkstemp(prefix="reviews_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(all_items, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, REVIEWS_FILE)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

def _public_url(local_path: str) -> str:
    # uploads/xxx.jpg → /static/xxx.jpg  (main.py에서 /static 마운트 필수)
    return "/static/" + os.path.basename(local_path)

# ----------------------- CRUD -----------------------

@router.get("/reviews")
def list_reviews(skip: int = 0, limit: int = 10):
    items = _read_all()
    items_sorted = sorted(items, key=lambda x: x.get("createdAt", ""), reverse=True)
    total = len(items_sorted)
    return {"items": items_sorted[skip:skip + limit], "total": total}

@router.get("/reviews/{rid}")
def get_review(rid: str):
    items = _read_all()
    for doc in items:
        if doc.get("_id") == rid:
            return doc
    raise HTTPException(404, "존재하지 않는 글")

@router.post("/reviews")
def create_review(
    username: str = Depends(get_current_username),   # ✅ 의존성 주입
    title: str = Form(...),
    body: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
):
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
        "_id": str(uuid4()),
        "title": title,
        "body": body,
        "images": image_urls,
        "authorId": username,
        "authorName": username,
        "createdAt": now,
        "updatedAt": now,
    }

    items = _read_all()
    items.append(doc)
    _atomic_write(items)
    return doc

@router.patch("/reviews/{rid}")
def update_review(
    rid: str,
    username: str = Depends(get_current_username),  
    title: Optional[str] = Form(None),
    body: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(default=None),
):
    items = _read_all()
    idx = next((i for i, d in enumerate(items) if d.get("_id") == rid), None)
    if idx is None:
        raise HTTPException(404, "존재하지 않는 글")

    doc = items[idx]
    if doc.get("authorId") != username:
        raise HTTPException(403, "작성자만 수정 가능")

    if title is not None:
        doc["title"] = title
    if body is not None:
        doc["body"] = body

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
        doc["images"] = new_urls

    doc["updatedAt"] = _now_iso()
    items[idx] = doc
    _atomic_write(items)
    return doc

@router.delete("/reviews/{rid}")
def delete_review(rid: str, username: str = Depends(get_current_username)):  # ✅
    items = _read_all()
    idx = next((i for i, d in enumerate(items) if d.get("_id") == rid), None)
    if idx is None:
        raise HTTPException(404, "존재하지 않는 글")

    doc = items[idx]
    if doc.get("authorId") != username:
        raise HTTPException(403, "작성자만 삭제 가능")

    del items[idx]
    _atomic_write(items)
    return {"ok": True}
