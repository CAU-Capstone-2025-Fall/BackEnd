import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient, UpdateOne

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
    createdImg: Optional[str] = None
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
    improve: Optional[str] = None
    extractedFeature: Optional[str] = None

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
    neuterYn: Optional[str] = Query(None, description="중성화 여부"),
    improve: Optional[str] = Query(None, description="개선 여부"),

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
    if neuterYn:
        query["neuterYn"] = neuterYn
    if improve:
        query["improve"] = improve
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


@router.get("/animal/notice-ids", response_model=List[str])
def get_notice_ids_with_improve():
    cursor = collection.find(
        {"improve": {"$in": [1, "1"]}},   # int 1 또는 string "1" 모두 매칭
        {"noticeNo": 1, "_id": 0}
    )
    return [doc["noticeNo"] for doc in cursor if "noticeNo" in doc]


router = APIRouter()

@router.put("/animal/update-many", response_model=dict)
def update_animals_by_notice(
    updates: List[Dict[str, str]] = Body(
        ..., 
        description="업데이트할 noticeNo-URL 쌍 리스트. 예: [{'noticeNo': '충남-부여-2025-00331', 'createdImg': 'https://...'}, ...]"
    )
):
    # 유효성 검사
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    # bulk operation 준비
    operations = []
    for item in updates:
        notice_no = item.get("noticeNo")
        created_img = item.get("createdImg")
        if not notice_no or not created_img:
            raise HTTPException(status_code=400, detail=f"Invalid item: {item}")
        operations.append(
            UpdateOne(
                {"noticeNo": notice_no},
                {"$set": {"createdImg": created_img}}
            )
        )

    # bulk 실행
    result = collection.bulk_write(operations, ordered=False)

    return {
        "matched": result.matched_count,
        "modified": result.modified_count,
        "acknowledged": result.acknowledged
    }
