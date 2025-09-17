import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()
# app = FastAPI()
router = APIRouter()

# DB 연결
try:
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri)
    db = client["testdb"]
    collection = db["abandoned_animals"]
    print ("db connected")
except Exception as e:
    print("db connection error:", e)

# Pydantic 모델
class Animal(BaseModel):
    desertionNo: str
    happenDt: Optional[str] = None
    happenPlace: Optional[str] = None
    kindFullNm: Optional[str] = None
    upKindCd: Optional[str] = None
    upKindNm: Optional[str] = None
    kindCd: Optional[str] = None
    kindNm: Optional[str] = None
    colorCd: Optional[str] = None
    age: Optional[str] = None
    weight: Optional[str] = None
    noticeNo: Optional[str] = None
    noticeSdt: Optional[str] = None
    noticeEdt: Optional[str] = None
    popfile1: Optional[str] = None
    popfile2: Optional[str] = None
    processState: Optional[str] = None
    sexCd: Optional[str] = None
    neuterYn: Optional[str] = None
    specialMark: Optional[str] = None
    sfeSoci: Optional[str] = None
    sfeHealth: Optional[str] = None
    etcBigo: Optional[str] = None
    careRegNo: Optional[str] = None
    careNm: Optional[str] = None
    careTel: Optional[str] = None
    careAddr: Optional[str] = None
    careOwnerNm: Optional[str] = None
    orgNm: Optional[str] = None
    updTm: Optional[str] = None
    vaccinationChk: Optional[str] = None
    healthChk: Optional[str] = None

def animal_serializer(animal) -> dict:
    animal = dict(animal)
    animal["id"] = str(animal["_id"])
    del animal["_id"]
    return animal
    
# CREATE
@router.post("/animal", response_model=dict)
def create_animal(animal: Animal):
    if collection.find_one({"desertionNo": animal.desertionNo}):
        raise HTTPException(status_code=400, detail="Animal already exists.")
    result = collection.insert_one(animal.model_dump())
    return {"id": str(result.inserted_id)}

# READ (조건에 따른 전체 or desertionNo로 조회)
@router.get("/animal", response_model=List[dict])
def get_animals(
    start_date: Optional[str] = Query(None, description="조회 시작 날짜 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="조회 끝 날짜 (YYYY-MM-DD)"),
    happen_place: Optional[str] = Query(None, description="발생 장소"),
    upkind_nm: Optional[str] = Query(None, description="대분류 이름"),
    kind_nm: Optional[str] = Query(None, description="세부 품종 이름"),
    sex_cd: Optional[str] = Query(None, description="성별 코드"),
    care_name: Optional[str] = Query(None, description="보호소 이름"),
    org_name: Optional[str] = Query(None, description="시군구 정보"),
    limit: int = 10,
    skip: int = 0
):
    query = {}
    if start_date and end_date:
        query["happenDt"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["happenDt"] = {"$gte": start_date}
    elif end_date:
        query["happenDt"] = {"$lte": end_date}
    if happen_place:
        query["happenPlace"] = happen_place
    if upkind_nm:
        query["upKindNm"] = upkind_nm
    if kind_nm:
        query["kindNm"] = kind_nm
    if sex_cd:
        query["sexCd"] = sex_cd
    if care_name:
        query["careNm"] = care_name
    if org_name:
        query["orgNm"] = org_name
    animals = list(collection.find(query).skip(skip).limit(limit))
    return [animal_serializer(animal) for animal in animals]

@router.get("/animal/{desertion_no}", response_model=dict)
def get_animal(desertion_no: str):
    animal = collection.find_one({"desertionNo": desertion_no})
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    return animal_serializer(animal)

# UPDATE
@router.put("/animal/{desertion_no}", response_model=dict)
def update_animal(desertion_no: str, animal: Animal):
    result = collection.update_one({"desertionNo": desertion_no}, {"$set": animal.model_dump()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Animal not found")
    updated = collection.find_one({"desertionNo": desertion_no})
    return animal_serializer(updated)

# DELETE
@router.delete("/animal/{desertion_no}", response_model=dict)
def delete_animal(desertion_no: str):
    result = collection.delete_one({"desertionNo": desertion_no})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Animal not found")
    return {"msg": f"Animal {desertion_no} deleted."}