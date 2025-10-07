import os
import json
from typing import List
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/userinfo", tags=["userinfo"])

DATA_DIR = "survey_data"
os.makedirs(DATA_DIR, exist_ok=True)

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

@router.post("/survey") # survey_data 폴더에 userId.txt 파일로 저장
async def save_survey(data: SurveyRequest, request: Request):
    filename = os.path.join(DATA_DIR, f"{data.userId}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(data.model_dump(), ensure_ascii=False, indent=2))
    print("설문 응답 저장:", data.model_dump())
    return {"success": True, "msg": "설문 저장 완료"}

@router.get("/survey/{userId}") # userId.txt 파일 읽어서 응답
async def get_survey(userId: str):
    filename = os.path.join(DATA_DIR, f"{userId}.txt")
    if not os.path.exists(filename):
        return {"success": False, "msg": "설문 응답이 없습니다."}
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"success": True, "data": data}