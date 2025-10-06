import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient, UpdateOne

load_dotenv()
# app = FastAPI()
router = APIRouter(tags=["Animal"])

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

def get_priority_score(animal):
    score = 0
    today = datetime.today().date()

    # #공고 종료일에 따른 가산점 <- 공고기간은 입양 동물과는 무관
    # notice_edt = datetime.strptime(animal["noticeEdt"], "%Y%m%d")
    # until_end = (notice_edt.date() - today).days
    # if until_end < 7:
    #     score += 50

    #입소 기간에 따른 가산점
    happen_dt = datetime.strptime(animal["happenDt"], "%Y%m%d")
    days = (today - happen_dt.date()).days
    days = days - 14 # 공고 기간 가산점 제외
    # 입소 기간이 길수록 가산점 증가
    if days >= 30:
        score += days
    if days >= 60:
        score += days
    if days > 90:
        score += days
    # 입소 후 3개월 이상인 경우 최대 200점 한도
    score += min(score, 200)

    #중성화 여부에 따른 가산점
    if animal.get("neuterYn") == "Y":
        score += 20
    
    # 건강 상태에 따른 가산점
    health = animal.get("specialMark", "").lower()
    if "불량" in health or "감염" in health or "병" in health:
        score += 50
    elif "양호" in health or "건강" in health:
        score += 20

    # 나이에 따른 가산점
    age_str = animal.get("age")
    if age_str:
        year_str = re.search(r"(\d{4})\(년생\)", age_str)
        year = int(year_str.group(1)) if year_str else None

        day_str = re.search(r"(\d+)일미만", age_str)
        day = int(day_str.group(1)) if day_str else None
    if day: # 1년 미만
        score += 50
    elif year:
        if datetime.today().year - year < 2:
            score += 30
        elif datetime.today().year - year < 5:
            score += 10

    # 사진 개선 여부에 따른 가산점 <- 개선 사진 효과 증대 목적
    if animal.get("improve") == "1":
        score += 50

    # 지역에 따른 가산점 추가 가능
    address = animal.get("careAddr", "")
    pattern = r'^(\S+도|\S+특별시|\S+광역시|\S+자치시|\S+시)\s+(\S+시|\S+군|\S+구)\s+(\S+구|\S+군|\S+동|\S+읍|\S+면)?'

    match = re.match(pattern, address)
    if match:
        sido = match.group(1)
        sigungu = match.group(2)
        gita = match.group(3) if match.group(3) else ""

    # if sido == "user_sido":
    #     score += 30
    #     if sigungu == "user_sigungu":
    #         score += 20
    #         if gita == "user_gita":
    #             score += 10

    # 추출된 특징에 따른 가산점 추가 가능
    # feature = animal.get("extractedFeature", "")
    # if "user_feature" in feature:
    #     score += 30

    return score

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

# READ 우선순위 알고리즘 적용
@router.get("/animal_priority", response_model=List[dict])
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

    animals = list(collection.find(query))
    animals.sort(key=get_priority_score, reverse=True)
    animals = animals[skip: skip + limit]
    for animal in animals:
        print(get_priority_score(animal))
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

@router.put("/animal/image/update", response_model=dict)
def update_image_by_desertion(data: dict = Body(...)):
    desertion_no = data.get("desertionNo")
    created_img = data.get("createdImg")

    if not desertion_no or not created_img:
        raise HTTPException(status_code=400, detail="Missing desertionNo or createdImg")

    try:
        uri = os.getenv("MONGODB_URI")
        client = MongoClient(uri)
        db = client["testdb"]
        collection = db["abandoned_animals"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}")

    result = collection.update_one(
        {"desertionNo": desertion_no},
        {"$set": {"createdImg": created_img}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Animal {desertion_no} not found")

    return {"msg": f"Image updated for {desertion_no}"}


class UpdateItem(BaseModel):
    noticeNo: str
    createdImg: str

# ✅ 리스트 자체를 또 하나의 모델로 감싸기
class UpdateRequest(BaseModel):
    updates: List[UpdateItem]

@router.put("/animal/update-many", response_model=dict)
def bulk_update_animals_by_notice(req: UpdateRequest):
    updates = req.updates
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    operations = [
        UpdateOne(
            {"noticeNo": item.noticeNo},
            {"$set": {"createdImg": item.createdImg}}
        )
        for item in updates
    ]

    result = collection.bulk_write(operations, ordered=False)
    return {
        "matched": result.matched_count,
        "modified": result.modified_count,
        "acknowledged": result.acknowledged,
    }