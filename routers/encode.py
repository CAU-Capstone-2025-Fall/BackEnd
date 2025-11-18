# routers/encode.py
import os
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from routers.inference import infer_lime, infer_student
from utils.gpt_summary_utils import generate_recommendations_text, generate_summary_text

load_dotenv()

router = APIRouter(prefix="/encode", tags=["Encode"])

# --------------------------------------------------------
# MongoDB 연결
# --------------------------------------------------------
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client[os.getenv("REPORT_DB", "pet_rec")]
collection = db["reports"]

# --------------------------------------------------------
# 저장 스키마
# --------------------------------------------------------
class ReportData(BaseModel):
    latent_vector: list | dict | None = None
    summary: str | None = None
    recommendations: str | None = None
    raw_input: dict | None = None
    lime: dict | None = None
    logits: list | None = None
    probability: float | None = None
    percentile: int | None = None
    timestamp: str | None = None


# --------------------------------------------------------
# 영어→한글 매핑
# --------------------------------------------------------
FIELD_MAP = {
    "age": "연령",
    "familyCount": "가족 구성원 수",
    "houseSize": "주택규모",
    "budget": "월평균 가구소득",

    "sex1": "성별_1",
    "sex2": "성별_2",

    "residenceType1": "주택형태_1",
    "residenceType2": "주택형태_2",
    "residenceType3": "주택형태_3",
    "residenceType4": "주택형태_4",

    "wantingPet": "향후 반려동물 사육의향",

    "job1": "화이트칼라",
    "job2": "블루칼라",
    "job7": "자영업",
    "job8": "비경제활동층",
    "job10": "기타",
}

def convert_to_korean_keys(A: dict) -> dict:
    converted = {}
    for eng, val in A.items():
        if eng not in FIELD_MAP:
            continue

        key = FIELD_MAP[eng]

        # 값 숫자로 변환
        if isinstance(val, str):
            try:
                val = float(val)
            except:
                pass

        converted[key] = val

    return converted


# --------------------------------------------------------
# Report 저장 함수
# --------------------------------------------------------
async def save_report(user_id: str, report: ReportData):
    collection.update_one(
        {"userId": user_id},
        {"$set": {"userId": user_id, "report": report.dict()}},
        upsert=True
    )


# --------------------------------------------------------
# POST /encode/{user_id}
# --------------------------------------------------------
@router.post("/{user_id}")
async def encodeA_and_save(user_id: str, features: dict):

    print("\n===== 🔥 encode 호출 =====")
    print("raw features:", features)

    feat_dict = convert_to_korean_keys(features)
    print("converted:", feat_dict)

    # ------------------------------------
    # 1) 모델 예측
    # ------------------------------------
    try:
        result = infer_student(feat_dict)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"infer_student 실패: {e}")

    prob = result["probability"]
    percentile = result.get("percentile", None)

    # ------------------------------------
    # 2) LIME 계산
    # ------------------------------------
    try:
        lime = infer_lime(pd.DataFrame([result["input_scaled"]]))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"infer_lime 실패: {e}")

    top5 = sorted(lime.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # ------------------------------------
    # 3) summary 생성용 clean input
    # ------------------------------------
    clean_input = {
        k: v
        for k, v in result["input_raw"].items()
        if not (str(v) == "0" or v == 0)
    }

    # ------------------------------------
    # 4) GPT 요약 생성
    # ------------------------------------
    try:
        summary = await generate_summary_text(prob, top5, clean_input)
    except Exception as e:
        traceback.print_exc()
        summary = None

    # ------------------------------------
    # 5) GPT 행동 추천 생성
    # ------------------------------------
    try:
        recommendations = await generate_recommendations_text(prob, top5, clean_input)
    except Exception as e:
        traceback.print_exc()
        recommendations = None

    # ------------------------------------
    # 6) ReportData 생성
    # ------------------------------------
    report = ReportData(
        raw_input=result["input_raw"],
        latent_vector=result["latent_vector"],
        logits=result["logits"],
        probability=prob,
        percentile=percentile,
        lime=lime,
        summary=summary,
        recommendations=recommendations,
        timestamp=datetime.utcnow().isoformat()
    )

    # ------------------------------------
    # 7) DB 저장
    # ------------------------------------
    await save_report(user_id, report)

    return {
        "success": True,
        "data": report.dict()
    }
