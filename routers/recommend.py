import os
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, FastAPI, Query, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from datetime import datetime
import re
import json
import numpy as np
from utils.location import location_score

router = APIRouter(prefix="/recommand", tags=["recommand"])

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI    = os.getenv("MONGODB_URI")
MODEL          = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]
survey_col = db["userinfo"]

app = FastAPI()

def get_embedding(text: str):
    resp = client_ai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def extract_species(natural_query: str):
    species_map = {
        "개": ["개", "강아지", "dog", "멍멍이"],
        "고양이": ["고양이", "냥이", "cat"],
        "기타": ["기타", "토끼", "햄스터", "기니피그", "고슴도치", "거북이", "새", "파충류"]  # 필요에 따라 확장
    }
    q = (natural_query or "").lower()
    found = set()
    for key, keywords in species_map.items():
        for word in keywords:
            if word in q:
                found.add(key)
                break
    return sorted(found)

def parse_size(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["대형","큼","큰 편","large"]): return "대형"
    if any(k in t for k in ["중간","중형","medium"]):       return "중형"
    if any(k in t for k in ["소형","작","small"]):          return "소형"
    return "알수없음"

def days_since(noticeSdt: Optional[str]) -> int:
    if not noticeSdt: return 0
    try:
        dt = datetime.strptime(noticeSdt, "%Y%m%d")
        return (datetime.now() - dt).days
    except Exception:
        return 0
    
# 우선순위 기반 노출 알고리즘을 추천에도 약간의 가중치로 적용
def priority_boost(doc: Dict[str, Any]) -> float:
    score = 0.0
    # 나이
    age_txt = doc.get("age", "") or ""
    m = re.search(r"(\d{4})", age_txt)
    if m:
        years = datetime.now().year - int(m.group(1))
        if years >= 7: score += 1.0
        elif years >= 4: score += 0.5
    # 크기
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    if a_size == "대형": score += 0.5
    # 건강/특수
    marks = " ".join([
        doc.get("specialMark","") or "",
        (doc.get("extractedFeature") or {}).get("noticeable_features","") or "",
        doc.get("healthChk","") or "",
    ])
    if any(w in marks for w in ["불량", "감염", "병", "장애", "실명", "결손", "오염"]):
        score += 1.0
    # 장기 체류
    if days_since(doc.get("noticeSdt")) >= 30: score += 0.5
    # 믹스
    if doc.get("kindNm") in ["믹스견", "한국 고양이"]: score += 0.25
    return score  # 0~3 근사

def is_generic_query(q: str) -> bool:
    if not q or len(q.strip()) < 4:
        return True
    q = q.lower()
    # 도메인 키워드(색/크기/털/성격/신체 등)
    keywords = [
        "개","강아지","dog","고양","cat","햄스터","토끼","파충","조류",
        "검정","흰","베이지","갈","회색","연한","진한",
        "소형","중형","대형","큰", "작은", "짧은 털","긴 털","장모","단모",
        "활발","차분","조용","사교","독립","애교","온순","귀여운",
        "귀","눈","꼬리","털","무늬","점박이","얼룩",
    ]
    if any(k in q for k in keywords):
        # 키워드는 있지만 “추천해줘/추천” 등만 있을 때 길이가 너무 짧으면 generic
        tokens = re.findall(r"[가-힣a-zA-Z0-9]+", q)
        return len(tokens) < 3
    # 키워드가 전혀 없으면 generic
    return True

def build_profile_text(ans: Dict[str, Any]) -> str:
    # 설문 응답을 요약 텍스트로 변환 → 임베딩 입력으로 사용
    if not ans: return ""
    fav = ", ".join(ans.get("favoriteAnimals", []) or [])
    pers = ", ".join(ans.get("preferredPersonality", []) or [])
    expect = ", ".join(ans.get("expectations", []) or [])
    curPets = ", ".join(ans.get("currentPets", []) or [])
    parts = [
        f"선호 종: {fav or '상관없음'}",
        f"선호 크기: {ans.get('preferredSize') or '상관없음'}",
        f"선호 성격: {pers or '상관없음'}",
        f"기대: {expect or '미지정'}",
        f"활동 수준: {ans.get('activityLevel') or '보통'}",
        f"하루 케어 시간: {ans.get('careTime') or '미지정'}",
        f"주거: {ans.get('residenceType') or '미지정'}, 평수: {ans.get('houseSize') or '미지정'}",
        f"가족 수: {ans.get('familyCount') or '미지정'}",
        f"알레르기: {ans.get('hasAllergy') or '없음'} / 대상: {ans.get('allergyAnimal') or '무'}",
        f"현재 반려동물: {curPets or '없음'}",
        f"주소: {ans.get('address') or '미기재'}",
        f"예산: {ans.get('budget') or '미기재'}",
    ]
    return " | ".join(parts)

def survey_constraints(ans: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    # 알레르기 <- 설문 기반 사전 필터링은 알레르기만 엄격히 적용
    if ans.get("hasAllergy") == "있음":
        aa = (ans.get("allergyAnimal") or "").lower()
        if "고양" in aa and doc.get("upKindNm") == "고양이": return False
        if ("강아지" in aa or "개" in aa) and doc.get("upKindNm") == "개": return False
    # 선택적 선호 종 강제 <- 설문에서 선택한 종 이외도 추천받고 싶을 수 있으므로 제외
    # fav = ans.get("favoriteAnimals") or []
    # if fav:
    #     want_dog = any(("강아지" in x) or ("개" in x) for x in fav)
    #     want_cat = any("고양" in x for x in fav)
    #     if doc.get("upKindNm") == "개" and not want_dog: return False
    #     if doc.get("upKindNm") == "고양이" and not want_cat: return False
    # 주거/평수 vs 대형 <- 너무 엄격할 수 있어 제외
    # pref_size = ans.get("preferredSize") or ""
    # a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    # small_house = ans.get("houseSize") in ["10평 미만", "10평 ~ 20평"]
    # apt = ans.get("residenceType") == "아파트"
    # if pref_size == "소형" and a_size == "대형": return False
    # if (small_house or apt) and a_size == "대형": return False
    return True

def size_match(pref: str, a_size: str) -> float:
    if pref in ("", "상관없음") or a_size == "알수없음": return 0.2
    if pref == a_size: return 1.0
    if (pref, a_size) in [("소형","중형"),("중형","소형"),("대형","중형"),("중형","대형")]:
        return 0.5
    return 0.0

def effort_hint(a_size: str) -> Tuple[float,float]:
    if a_size == "대형": return (0.9, 0.8) # 활동, 케어
    if a_size == "중형": return (0.6, 0.5)
    if a_size == "소형": return (0.4, 0.3)
    return (0.5, 0.5)

def compat_score(ans: Dict[str, Any], doc: Dict[str, Any]) -> float:
    if not ans: return 0.0
    score, wsum = 0.0, 0.0
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))

    # 크기 선호
    w = 0.35
    score += w * size_match(ans.get("preferredSize",""), a_size); wsum += w

    # 활동/케어 적합
    desired_active = {"매우 활발함":0.9,"보통":0.6,"주로 실내 생활":0.3}.get(ans.get("activityLevel"),0.6)
    care_time = {"10분 이하":0.2,"30분":0.4,"1시간":0.6,"2시간 이상":0.9}.get(ans.get("careTime"),0.5)
    a_act, a_care = effort_hint(a_size)
    w = 0.25
    act_match = 1 - min(1.0, abs(desired_active - a_act))
    care_match = 1 - min(1.0, abs(care_time - a_care))
    score += w * ((act_match + care_match)/2); wsum += w

    # 주거/평수 적합
    small_house = ans.get("houseSize") in ["10평 미만","10평 ~ 20평"]
    apt = ans.get("residenceType") == "아파트"
    house_ok = 1.0
    if (small_house or apt) and a_size == "대형": house_ok = 0.1
    elif a_size == "중형" and (small_house or apt): house_ok = 0.6
    w = 0.2
    score += w * house_ok; wsum += w

    # 선호 성격(간단 매칭)
    pref_pers = set(ans.get("preferredPersonality") or [])
    persona_sources = [doc.get("specialMark") or "", ]
    persona_text = " ".join(persona_sources).lower()

    TEMPERAMENT_MAP = {
        "차분": ["차분", "조용", "온순", "침착", "순함", "순", "평온", "차분함"],
        "활발": ["활발", "에너지", "신남", "뛰", "활기", "명랑", "발랄"],
        "독립": ["독립", "자립", "혼자 잘", "스스로", "독립적"],
        "애교": ["애교", "교감", "스킨십", "사람 좋아", "친밀", "붙임성", "친화"],
        "예민": ["예민", "민감", "신경질", "경계", "조심스러움", "날카로움"],
    }

    # 사용자 선호에서 '상관없음' 처리
    if pref_pers and pref_pers == {"상관없음"}:
        persona_score = 0.5  # 중립
    else:
        persona_score = 0.2  # 기본값
        if pref_pers:
            hits = 0.0
            for pref in pref_pers:
                # pref가 카테고리에 없으면 건너뜀
                if pref not in TEMPERAMENT_MAP and pref != "상관없음":
                    continue

                # 일반 선호(차분/활발/독립/애교)
                if pref in TEMPERAMENT_MAP:
                    if any(k in persona_text for k in TEMPERAMENT_MAP[pref]):
                        hits += 1.0
                    else:
                        # ‘차분’을 원했지만 ‘예민’ 특성만 있는 경우 부분 점수 부여
                        if pref == "차분" and any(k in persona_text for k in TEMPERAMENT_MAP["예민"]):
                            hits -= 0.3 

            effective_pref_count = max(1, len([p for p in pref_pers if p in TEMPERAMENT_MAP]))
            persona_score = min(1.0, hits / effective_pref_count)

    # 가중치 적용
    w = 0.2
    score += w * persona_score
    wsum += w

    return max(0.0, min(1.0, score / (wsum or 1)))

class RecommendRequest(BaseModel):
    natural_query: str
    limit: int = 3

class HybridRequest(BaseModel):
    natural_query: str
    limit: int = 6
    user_id: Optional[str] = None
    use_survey: bool = True

@router.post("")
def recommend_animals(body: RecommendRequest):
    # 1. 사용자 쿼리 임베딩
    try:
        user_emb = get_embedding(body.natural_query)
    except Exception as e:
        raise HTTPException(500, f"임베딩 생성 오류: {e}")

    # 2. 모든 동물 문서 순회, (extractedFeature 등으로 임베딩)
    species_list = extract_species(body.natural_query)
    query_filter: Dict[str, Any] = {}
    if species_list:
        # 다중 종 필터 적용
        if len(species_list) == 1:
            query_filter = {"upKindNm": species_list[0]}
        else:
            query_filter = {"upKindNm": {"$in": species_list}}
    candidates = []
    for doc in collection.find(query_filter):
        animal_emb = np.array(doc.get("embedding", []))
        score = cosine_similarity(user_emb, animal_emb)
        candidates.append((score, doc))

    topn = sorted(candidates, key=lambda x: x[0], reverse=True)[:body.limit]
    return [
        {
            "score": round(s, 4),
            "desertionNo": doc.get("desertionNo"),
            "kindFullNm": doc.get("kindFullNm"),
            # 필요한 정보 추가
        }
        for (s, doc) in topn if s > 0.001 
    ]

@router.post("/hybrid")
def recommend_hybrid(body: HybridRequest):
    # 1) 사용자 쿼리/프로필 임베딩
    try:
        q = (body.natural_query or "").strip()
        q_emb = get_embedding(q) if q else None
    except Exception as e:
        raise HTTPException(500, f"임베딩 생성 오류: {e}")

    survey = {}
    if body.use_survey and body.user_id:
        sdoc = survey_col.find_one({"userId": body.user_id}, sort=[("_id",-1)])
        if sdoc:
            survey = sdoc.get("answers") or sdoc

    profile_text = build_profile_text(survey) if survey else ""
    p_emb = get_embedding(profile_text) if profile_text else None

    generic = is_generic_query(q)
    # 쿼리/설문 유사도 결합 비율
    alpha = 0.3 if generic else 0.7  # generic일수록 설문 비중 증가
    w_sim, w_comp, w_prio, w_loc = (0.35, 0.55, 0.1, 0.1) if generic else (0.6, 0.3, 0.1, 0.1)

    # 2) 후보 생성
    species_list = extract_species(q)
    base_filter: Dict[str, Any] = {}
    if species_list:
        if len(species_list) == 1:
            base_filter["upKindNm"] = species_list[0]
        else:
            base_filter["upKindNm"] = {"$in": species_list}
    
    candidates: List[Tuple[float, Dict[str,Any]]] = []
    for doc in collection.find(base_filter):
        a_emb = np.array(doc.get("embedding") or [], dtype=np.float32)
        if a_emb.size == 0: continue

        sim_q = cosine_similarity(q_emb, a_emb) if q_emb is not None else 0.0
        sim_p = cosine_similarity(p_emb, a_emb) if p_emb is not None else 0.0
        sim_mix = (alpha * sim_q) + ((1 - alpha) * sim_p) if (q_emb is not None or p_emb is not None) else 0.0
        if sim_mix <= 0: continue
        candidates.append((sim_mix, doc))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    pool = candidates[: max(100, body.limit * 10)]

    # 3) 설문 제약 필터
    filtered = []
    for sim, doc in pool:
        if survey and not survey_constraints(survey, doc):
            continue
        filtered.append((sim, doc))
    if not filtered:
        filtered = pool

    # 4) 호환/우선노출/위치기반 계산 + 최종 재랭킹
    scored: List[Tuple[float, Dict[str,Any], Dict[str,float]]] = []
    user_addr = survey.get("address") if survey else None
    for sim, doc in filtered:
        comp = compat_score(survey, doc) if survey else 0.0
        prio = priority_boost(doc)
        loc = location_score(user_addr, doc.get("careAddr"))
        final = w_sim*sim + w_comp*comp + w_prio*(prio/3.0) + w_loc*loc  # prio 0~3 → 0~1 정규화
        scored.append((final, doc, {"sim":sim, "comp":comp, "prio":prio, "loc":loc}))

    scored.sort(key=lambda x: x[0], reverse=True)
    topn = scored[: body.limit]

    return [
        {
            "final": round(s,4),
            "sim": round(meta["sim"],4),
            "compat": round(meta["comp"],4),
            "priority": round(meta["prio"],3),
            "location": round(meta["loc"],4),
            "desertionNo": d.get("desertionNo"),
            "kindFullNm": d.get("kindFullNm"),
            "upKindNm": d.get("upKindNm"),
            "age": d.get("age"),
            "careAddr": d.get("careAddr"),
            "extractedFeature": d.get("extractedFeature"),
        }
        for (s, d, meta) in topn
    ]