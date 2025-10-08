import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()

router = APIRouter(prefix="/userinfo", tags=["userinfo"])

# MongoDB 연결
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["testdb"]             # 👉 실제 DB 이름으로 변경 가능
userinfo_col = db["userinfo"]     # 새 컬렉션 (없으면 자동 생성)

# -------------------- 모델 정의 --------------------
class SurveyRequest(BaseModel):
    userId: str
    address: str
    residenceType: str
    hasPetSpace: str
    familyCount: str
    hasChildOrElder: str
    dailyHomeTime: str
    hasAllergy: str
    allergyAnimal: str
    activityLevel: str
    expectations: List[str]
    favoriteAnimals: List[str]
    preferredSize: str
    preferredPersonality: List[str]
    careTime: str
    budget: str
    specialEnvironment: str
    additionalNote: str


# -------------------- CRUD --------------------
@router.post("/survey")
async def save_survey(data: SurveyRequest):
    """
    설문 응답 저장 (userId 기준으로 upsert)
    """
    doc = data.model_dump()
    userinfo_col.update_one(
        {"userId": data.userId},
        {"$set": doc},
        upsert=True  # 이미 있으면 갱신, 없으면 새로 삽입
    )
    print("✅ 설문 응답 저장:", data.userId)
    return {"success": True, "msg": "설문 저장 완료"}


@router.get("/survey/{userId}")
async def get_survey(userId: str):
    """
    userId 기준 설문 조회
    """
    doc = userinfo_col.find_one({"userId": userId}, {"_id": 0})
    if not doc:
        return {"success": False, "msg": "설문 응답이 없습니다."}
    return {"success": True, "data": doc}
