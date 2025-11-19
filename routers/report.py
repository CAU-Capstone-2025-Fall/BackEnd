# router/report.py
import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()

router = APIRouter(prefix="/report", tags=["Report"])

# -----------------------------
# MongoDB 연결
# -----------------------------
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("REPORT_DB", "pet_rec")]   # testdb → pet_rec로 통일
reports = db["reports"]

# -----------------------------
# 데이터 모델
# -----------------------------
class ReportData(BaseModel):
    latent_vector: list | dict | None = None
    summary: str | None = None
    raw_input: dict | None = None
    timestamp: str | None = None


# -----------------------------
# Report 저장 (Upsert)
# -----------------------------
@router.post("/{user_id}")
async def save_report(user_id: str, data: ReportData):
    """
    userId 기준으로 리포트 1개만 저장. 있으면 덮어쓰기.
    """
    doc = data.model_dump()

    reports.update_one(
        {"userId": user_id},
        {"$set": {"report": doc}},
        upsert=True
    )

    return {"success": True, "msg": "리포트 저장 완료"}


# -----------------------------
# Report 조회
# -----------------------------
@router.get("/{user_id}")
async def get_report(user_id: str):
    """
    userId 기준 리포트 조회. 
    항상 1개만 존재 (upsert 구조).
    """
    doc = reports.find_one({"userId": user_id}, {"_id": 0})

    if not doc or "report" not in doc:
        return {"success": False, "msg": "리포트 없음"}

    return {"success": True, "data": doc["report"]}
