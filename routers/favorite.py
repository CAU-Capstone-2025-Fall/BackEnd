# routers/favorites.py
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pymongo import MongoClient
from routers.login import get_current_username

load_dotenv()

router = APIRouter(tags=["favorite"])

# MongoDB 연결
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["testdb"]               # 👉 실제 DB 이름으로 변경 가능
favorites_col = db["favorites"]     # 새 컬렉션 생성됨 (없으면 자동 생성)

# ----------------------------------------------------------
# 기본 구조:
# {
#   "username": "minseok",
#   "favorites": ["448543202500133", "444457202500540", ...]
# }
# ----------------------------------------------------------

@router.get("/favorite", response_model=Dict[str, Any])
def get_favorites(username: str = Depends(get_current_username)):
    """사용자의 즐겨찾기 목록 반환"""
    doc = favorites_col.find_one({"username": username})
    if not doc:
        favorites_col.insert_one({"username": username, "favorites": []})
        return {"ids": []}
    return {"ids": doc.get("favorites", [])}


@router.post("/favorite/{animal_id}", response_model=Dict[str, Any])
def add_favorite(animal_id: str, username: str = Depends(get_current_username)):
    """즐겨찾기 추가"""
    doc = favorites_col.find_one({"username": username})
    if not doc:
        favorites_col.insert_one({"username": username, "favorites": [animal_id]})
        return {"ids": [animal_id]}

    favs = doc.get("favorites", [])
    if animal_id not in favs:
        favs.append(animal_id)
        favorites_col.update_one({"username": username}, {"$set": {"favorites": favs}})
    return {"ids": favs}


@router.delete("/favorite/{animal_id}", response_model=Dict[str, Any])
def remove_favorite(animal_id: str, username: str = Depends(get_current_username)):
    """즐겨찾기 제거"""
    doc = favorites_col.find_one({"username": username})
    if not doc:
        favorites_col.insert_one({"username": username, "favorites": []})
        return {"ids": []}

    favs = [x for x in doc.get("favorites", []) if x != animal_id]
    favorites_col.update_one({"username": username}, {"$set": {"favorites": favs}})
    return {"ids": favs}
