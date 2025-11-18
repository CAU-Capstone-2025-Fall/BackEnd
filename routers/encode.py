# routers/encode.py
import asyncio
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
from utils.interaction_utils import compute_interaction_sets

load_dotenv()

router = APIRouter(prefix="/encode", tags=["Encode"])

# --------------------------------------------------------
# MongoDB
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
    interaction: list | None = None
    timestamp: str | None = None


# --------------------------------------------------------
# 영어→한글
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
        if eng in FIELD_MAP:
            key = FIELD_MAP[eng]
            try:
                val = float(val)
            except:
                pass
            converted[key] = val
    return converted


# --------------------------------------------------------
# Interaction Feature Groups
# --------------------------------------------------------
FEATURE_GROUPS = {
    "연령": ["연령"],
    "가족": ["가족 구성원 수"],
    "주택규모": ["주택규모"],
    "소득": ["월평균 가구소득"],

    "성별": ["성별_1", "성별_2"],
    "주택형태": ["주택형태_1","주택형태_2","주택형태_3","주택형태_4"],
    "직업": ["화이트칼라","블루칼라","자영업","비경제활동층","기타"],
}


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
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "infer_lime 실패")

    top5 = sorted(lime.items(), key=lambda x: abs(x[1]), reverse=True)[:5]


    # ------------------------------------
    # 3) Interaction 계산 (GPT보다 먼저)
    # ------------------------------------
    def f_local(d):
        return infer_student(d)["probability"]

    try:
        interaction_top3 = compute_interaction_sets(
            result["input_raw"],
            f_local,
            FEATURE_GROUPS,
            top_k=3
        )
    except Exception:
        traceback.print_exc()
        interaction_top3 = None

    print("INTERACTIONS:", interaction_top3)


    # ------------------------------------
    # 4) GPT 요약/추천 (interaction 포함)
    # ------------------------------------
    clean_input = {
        k: v for k, v in result["input_raw"].items()
        if not (str(v) == "0" or v == 0)
    }

    try:
        summary_task = generate_summary_text(prob, top5, interaction_top3, clean_input)
        rec_task = generate_recommendations_text(prob, top5, interaction_top3, clean_input)
        summary, recommendations = await asyncio.gather(summary_task, rec_task)
    except Exception:
        traceback.print_exc()
        summary = None
        recommendations = None


    # ------------------------------------
    # 5) ReportData 구성
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
        interaction=interaction_top3,
        timestamp=datetime.utcnow().isoformat(),
    )

    await save_report(user_id, report)

    return {
        "success": True,
        "data": report.dict()
    }
