# routers/favorites.py
from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
from threading import RLock
from typing import List, Dict, Any
from routers.login import get_current_username

router = APIRouter(tags=["favorite"])

BASE_DIR = Path("user_favorites")
BASE_DIR.mkdir(exist_ok=True)
LOCK = RLock()

def _file_of(user: str) -> Path:
    return BASE_DIR / f"{user}.txt"

def _read_ids(user: str) -> List[str]:
    p = _file_of(user)
    if not p.exists():
        return []
    with LOCK:
        raw = p.read_text(encoding="utf-8")
    parts = raw.replace(",", "\n").splitlines()
    ids: List[str] = []
    seen = set()
    for s in parts:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            ids.append(s)
    return ids

def _write_ids(user: str, ids: List[str]) -> None:
    p = _file_of(user)
    text = "\n".join(ids) + ("\n" if ids else "")
    with LOCK:
        p.write_text(text, encoding="utf-8")

@router.get("/favorite", response_model=Dict[str, Any])
def get_favorites(username: str = Depends(get_current_username)):
    ids = _read_ids(username)
    return {"ids": ids}

@router.post("/favorite/{animal_id}", response_model=Dict[str, Any])
def add_favorite(animal_id: str, username: str = Depends(get_current_username)):
    ids = _read_ids(username)
    if animal_id not in ids:
        ids.append(animal_id)
        _write_ids(username, ids)
    return {"ids": ids}

@router.delete("/favorite/{animal_id}", response_model=Dict[str, Any])
def remove_favorite(animal_id: str, username: str = Depends(get_current_username)):
    ids = [x for x in _read_ids(username) if x != animal_id]
    _write_ids(username, ids)
    return {"ids": ids}
