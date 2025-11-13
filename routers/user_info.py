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
    age: str
    sex: str
    job: str
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
    petHistory: str
    currentPets: List[str]
    houseSize: str
    wantingPet: str

class ProcessedSurveyData(BaseModel):
    userId: str
    age: str
    familyCount: str
    houseSize: str
    budget: str
    sex1: str
    sex2: str
    residenceType1: str
    residenceType2: str
    residenceType3: str
    residenceType4: str
    job1: str
    job10: str
    job2: str
    job3: str
    job4: str
    job5: str
    job6: str
    job7: str
    job8: str
    job9: str
    petHistory1: str
    petHistory2: str
    petHistory3: str
    wantingPet: str

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

# 추천 시스템에 사용할 데이터로 가공하여 반환
@router.get("/survey/processed/{userId}")
async def get_processed_survey(userId: str):
    """
    userId 기준 설문 조회 (가공된 데이터)
    """
    doc = userinfo_col.find_one({"userId": userId}, {"_id": 0})
    if not doc:
        return {"success": False, "msg": "설문 응답이 없습니다."}

    # 가공된 데이터로 변환
    processed_data = get_A_processed(SurveyRequest(**doc))

    return {"success": True, "data": processed_data}

def get_A_processed(data: SurveyRequest):
    A_processed = ProcessedSurveyData(
        userId = data.userId,
        age = '',
        familyCount = '',
        houseSize = '',
        budget = '',
        sex1 = '0',
        sex2 = '0',
        residenceType1 = '0',
        residenceType2 = '0',
        residenceType3 = '0',
        residenceType4 = '0',
        job1 = '0',
        job2 = '0',
        job3 = '0',
        job4 = '0',
        job5 = '0',
        job6 = '0',
        job7 = '0',
        job8 = '0',
        job9 = '0',
        job10 = '0',
        petHistory1 = '0',
        petHistory2 = '0',
        petHistory3 = '0',
        wantingPet = ''
    )
    A_processed.age = data.age
    A_processed.familyCount = data.familyCount
    match data.houseSize:
        case '10평 미만':
            A_processed.houseSize = '5'
        case '10평 ~ 20평':
            A_processed.houseSize = '15'
        case '20평 ~ 30평':
            A_processed.houseSize = '25'
        case '30평 ~ 40평':
            A_processed.houseSize = '35'
        case '40평 ~ 50평':
            A_processed.houseSize = '45'
        case '50평 이상':
            A_processed.houseSize = '60'
    match data.budget:
        case '100만원 미만':
            A_processed.budget = '50'
        case '100만원 ~ 199만원':
            A_processed.budget = '150'
        case '200만원 ~ 299만원':
            A_processed.budget = '250'
        case '300만원 ~ 399만원':
            A_processed.budget = '350'
        case '400만원 ~ 499만원':
            A_processed.budget = '450'
        case '500만원 ~ 599만원':
            A_processed.budget = '550'
        case '600만원 ~ 699만원':
            A_processed.budget = '650'
        case '700만원 이상':
            A_processed.budget = '750'
    if (data.sex == '남성') :
      A_processed.sex1 = '1'
    elif (data.sex == '여성') :
      A_processed.sex2 = '1'
    match data.residenceType:
      case '아파트':
        A_processed.residenceType1 = '1'
      case '단독/다가구 주택':
        A_processed.residenceType2 = '1'
      case '연립/빌라/다세대 주택':
        A_processed.residenceType3 = '1'
      case '기타':
        A_processed.residenceType4 = '1'
    match data.job:
      case '경영/관리직':
        A_processed.job1 = '1'
      case '전문직':
        A_processed.job2 = '1'
      case '사무직':
        A_processed.job3 = '1'
      case '전문기술직':
        A_processed.job4 = '1'
      case '판매/서비스직':
        A_processed.job5 = '1'
      case '단순노무/생산/단순기술직':
        A_processed.job6 = '1'
      case '자영업':
        A_processed.job7 = '1'
      case '주부':
        A_processed.job8 = '1'
      case '학생':
        A_processed.job9 = '1'
      case '기타':
        A_processed.job10 = '1'
    match data.petHistory:
      case '현재 반려동물을 키우고 있다':
        A_processed.petHistory1 = '1'
      case '과거에는 키웠으나 현재는 키우고 있지 않다':
        A_processed.petHistory2 = '1'
      case '반려동물을 키운 적 없다':
        A_processed.petHistory3 = '1'
    match data.wantingPet:
      case '전혀 의향이 없다':
        A_processed.wantingPet = '0.2'
      case '별로 의향이 없다':
        A_processed.wantingPet = '0.4'
      case '보통이다':
        A_processed.wantingPet = '0.6'
      case '다소 의향이 있다':
        A_processed.wantingPet = '0.8'
      case '매우 의향이 있다':
        A_processed.wantingPet = '1.0'
      case _:
        A_processed.wantingPet = '0'

    return A_processed
