import os
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from datetime import datetime
import re
import numpy as np
from functools import lru_cache
from utils.location import location_score
from utils.filed_embedding import user_to_4texts
import json
from pathlib import Path
from utils.train_ltr_model import extract_features_for_ltr

router = APIRouter(prefix="/recommand", tags=["recommand"])

KEYWORD_EMBEDDINGS: Dict[str, np.ndarray] = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "keyword_embeddings.json"
try:
    with EMB_PATH.open("r", encoding="utf-8") as f:
        _raw = json.load(f)
        KEYWORD_EMBEDDINGS = {
            k: np.array(v, dtype=np.float32) for k, v in _raw.items()
        }
    print(f"[EMB] loaded {len(KEYWORD_EMBEDDINGS)} precomputed keyword embeddings from {EMB_PATH}")
except FileNotFoundError:
    print(f"[EMB] keyword_embeddings.json not found at {EMB_PATH}; keyword embeddings will call API on demand")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI    = os.getenv("MONGODB_URI")
EMBED_MODEL    = "text-embedding-3-small"

# LTR_MODEL = None
# MODEL_PATH = "ltr_model.pkl" # 학습 스크립트가 생성한 모델 파일
# try:
#     LTR_MODEL = joblib.load(MODEL_PATH)
#     print(f"[LTR] 랭킹 모델 '{MODEL_PATH}' 로드 성공.")
# except FileNotFoundError:
#     print(f"[LTR][WARN] LTR 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. API가 비활성화될 수 있습니다.")
# except Exception as e:
#     print(f"[LTR][ERROR] 모델 로드 중 오류 발생: {e}")

client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]
survey_col = db["userinfo"]
fav_col = db["favorites"]

app = FastAPI()

def merge_similarity_scores(
    base_mix: float,
    sim_f: float,
    field_match_score: float,
) -> float:
    """
    세 가지 점수(base, favorite, field)를 규칙에 따라 병합:
    - 0이면 영향에서 제외
    - 표준편차가 작으면 단순 평균
    - 표준편차가 큰 경우:
      * Outlier 없음(또는 모호): 
        - 가장 높은 점수가 base면 (전체 평균 + base)/2
        - 그 외에는 전체 평균
      * Outlier 존재:
        - outlier가 가장 작은 값일 때:
          · 그 값이 base면: 나머지 둘 평균과 base의 평균
          · 그 외: 가장 작은 값 무시, 나머지 둘 평균
        - outlier가 가장 큰 값일 때:
          · 그 값이 favorite면: 무시, 나머지 둘 중 더 큰 값
          · 그 외: 전체 평균과 최대값의 평균
    """

    values = {
        "base": float(base_mix),
        "fav": float(sim_f),
        "field": float(field_match_score),
    }

    # 0 이하 값은 영향에서 제외
    non_zero = {name: v for name, v in values.items() if v > 0.0}

    if not non_zero:
        return 0.0

    # 유효 점수가 1개면 그대로 사용
    if len(non_zero) == 1:
        return next(iter(non_zero.values()))

    # 유효 점수가 2개면 단순 평균
    if len(non_zero) == 2:
        return sum(non_zero.values()) / 2.0

    names = list(non_zero.keys())       # ["base","fav","field"] 
    vals = np.array([non_zero[n] for n in names], dtype=float)

    mean = float(vals.mean())
    std = float(vals.std(ddof=0))

    # 1) 표준편차가 작으면 그냥 평균
    SMALL_STD = 0.03  
    if std < SMALL_STD:
        return mean

    # 2) outlier 탐지
    #    z-score가 임계값보다 크면 outlier로 간주
    outlier_idx = []
    if std > 0:
        z = np.abs(vals - mean) / std
        Z_THRESHOLD = 1.2  
        outlier_idx = [i for i, zi in enumerate(z) if zi > Z_THRESHOLD]

    # outlier가 정확히 1개일 때만 "Outlier 존재" 케이스로 처리
    if len(outlier_idx) == 1:
        oi = outlier_idx[0]
        v_out = vals[oi]
        name_out = names[oi]
        min_val = float(vals.min())
        max_val = float(vals.max())

        # -------- outlier가 가장 작은 값인 경우 --------
        if v_out == min_val:
            if name_out == "base":
                # "가장 작은게 베이스 값인 경우, 나머지 두 값의 평균+베이스와의 평균"
                others = [vals[i] for i in range(3) if i != oi]
                others_mean = float(sum(others) / len(others))
                return (others_mean + values["base"]) / 2.0
            else:
                # "가장 작은값 무시. 나머지 값의 평균 구함"
                others = [vals[i] for i in range(3) if i != oi]
                return float(sum(others) / len(others))

        # -------- outlier가 가장 큰 값인 경우 --------
        if v_out == max_val:
            if name_out == "fav":
                # "가장 큰게 즐겨찾기인 경우, 무시. 나머지 값의 더 큰 값으로 결정"
                others = [vals[i] for i in range(3) if i != oi]
                return float(max(others))
            else:
                # "이외의 경우에는 세값의 평균과 가장 큰 값의 퍙균"
                return (mean + max_val) / 2.0

        # 이론상 여기 올 일은 거의 없지만, 이상 케이스는 그냥 평균으로
        return mean

    # 3) outlier가 없거나, 여러 개라서 모호한 경우
    max_val = float(vals.max())
    idx_max = int(vals.argmax())
    name_max = names[idx_max]

    if name_max == "base":
        # "가장 높은 점수가 베이스라면 세 값의 평균과의 베이스와의 평균 구하기."
        return (mean + values["base"]) / 2.0
    else:
        # "이외에는, 세 값의 평균 구하기"
        return mean

def safe_round(x: Any, ndigits: int = 4, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return default


def get_embedding(text: str) -> np.ndarray:
    resp = client_ai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Batch embedding using the OpenAI client. Returns list of numpy arrays in same order.
    Uses a simple retry on failure.
    """
    if not texts:
        return []
    # remove empty strings but preserve positions by mapping
    orig_idx = []
    req_texts = []
    for i, t in enumerate(texts):
        s = (t or "").strip()
        if s == "":
            orig_idx.append((i, None))
        else:
            orig_idx.append((i, s))
            req_texts.append(s)
    # if no real texts, return zeros
    if len(req_texts) == 0:
        return [np.zeros((1536,), dtype=np.float32) for _ in texts]

    # call embeddings API in one batch (handle simple retry/backoff)
    for attempt in range(5):
        try:
            resp = client_ai.embeddings.create(model=EMBED_MODEL, input=req_texts)
            vectors = [np.array(r.embedding, dtype=np.float32) for r in resp.data]
            break
        except Exception as e:
            wait = (2 ** attempt) * 0.5
            if attempt == 4:
                raise HTTPException(500, f"임베딩 생성 오류(재시도 후 실패): {e}")
            else:
                # lightweight sleep
                import time
                time.sleep(wait)
    # reconstruct full list preserving empty positions
    out = [None] * len(texts)
    j = 0
    for i, maybe in orig_idx:
        if maybe is None:
            out[i] = np.zeros((1536,), dtype=np.float32)
        else:
            out[i] = vectors[j]
            j += 1
    return out

@lru_cache(maxsize=1024)
def _cached_text_embedding(text: str) -> np.ndarray:
    return get_embedding(text)


@lru_cache(maxsize=256)
def _get_keyword_embedding(keyword: str) -> np.ndarray:
    kw = (keyword or "").strip()
    if not kw:
        if KEYWORD_EMBEDDINGS:
            dim = len(next(iter(KEYWORD_EMBEDDINGS.values())))
        else:
            dim = 1536  
        return np.zeros(dim, dtype=np.float32)

    if kw in KEYWORD_EMBEDDINGS:
        return KEYWORD_EMBEDDINGS[kw]

    print(f"[EMB][WARN] keyword '{kw}' not in precomputed KEYWORD_EMBEDDINGS, calling API")
    return get_embedding(kw)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _semantic_has_keywords(
    text: Optional[str],
    keywords: List[str],
    threshold: float = 0.60,
) -> bool:
    """
    text 전체와 각 keyword의 임베딩 코사인 유사도로 의미 기반 매칭.
    (단순 substring이 안 잡을 때 백업 용도)
    """
    if not text or not keywords:
        return False

    try:
        text_emb = _cached_text_embedding(text)
    except Exception:
        return False

    for kw in keywords:
        if not kw:
            continue
        kw_emb = _get_keyword_embedding(kw)
        sim = cosine_similarity(text_emb, kw_emb)
        if sim >= threshold:
            return True
    return False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -------------------------
# 도메인 파싱 / 추론 함수들
# -------------------------

def extract_species(natural_query: str) -> List[str]:
    """
    쿼리에서 종(개/고양이/기타) 추출.
    1차: substring
    2차: 쿼리 vs 각 keyword 의미 기반 유사도
    """
    species_map = {
        "개": ["개", "강아지", "dog", "멍멍이"],
        "고양이": ["고양이", "냥이", "cat"],
        "기타": ["기타", "토끼", "햄스터", "기니피그", "고슴도치", "거북이", "새", "파충류"],
    }
    q = (natural_query or "").lower()
    found = set()

    # 1차: 단순 substring 매칭
    for key, keywords in species_map.items():
        for word in keywords:
            if word in q:
                found.add(key)
                break

    if found:
        return sorted(found)

    # 2차: 의미 기반 매칭 (쿼리 vs 각 키워드 임베딩)
    try:
        q_emb = _cached_text_embedding(q)
    except Exception:
        return sorted(found)

    THRESHOLD = 0.60
    for key, keywords in species_map.items():
        for word in keywords:
            kw_emb = _get_keyword_embedding(word)
            sim = cosine_similarity(q_emb, kw_emb)
            if sim >= THRESHOLD:
                found.add(key)
                break

    return sorted(found)


def survey_preferred_species(ans: Dict[str, Any]) -> List[str]:
    """
    설문 favoriteAnimals에서 종 추출해서 upKindNm 값으로 변환.
    """
    if not ans:
        return []
    favs = ans.get("favoriteAnimals") or []
    species = set()
    for item in favs:
        t = str(item).lower()
        if "개" in t or "dog" in t or "강아지" in t:
            species.add("개")
        elif "고양" in t or "cat" in t or "냥이" in t:
            species.add("고양이")
        else:
            species.add("기타")
    return sorted(species)


def parse_size(text: str) -> str:
    """
    크기 텍스트를 대/중/소형으로 매핑.
    1차 substring, 2차 의미 기반.
    """
    t = (text or "").lower()

    # 1차: 단순 substring 매칭
    if any(k in t for k in ["대형","큼","큰","large"]):   return "대형"
    if any(k in t for k in ["중간","중형","중","medium"]): return "중형"
    if any(k in t for k in ["소형","작","small"]):        return "소형"

    # 2차: 의미 기반 매칭
    if not t.strip():
        return "알수없음"

    THRESHOLD = 0.60
    try:
        t_emb = _cached_text_embedding(t)
    except Exception:
        return "알수없음"

    LARGE_WORDS  = ["대형", "큰 편", "size large", "라지", "big dog", "big cat"]
    MEDIUM_WORDS = ["중형", "보통 크기", "medium size", "middle size"]
    SMALL_WORDS  = ["소형", "작은 편", "small size", "little dog", "little cat"]

    def any_sim(words: List[str]) -> bool:
        for w in words:
            kw_emb = _get_keyword_embedding(w)
            sim = cosine_similarity(t_emb, kw_emb)
            if sim >= THRESHOLD:
                return True
        return False

    if any_sim(LARGE_WORDS):
        return "대형"
    if any_sim(MEDIUM_WORDS):
        return "중형"
    if any_sim(SMALL_WORDS):
        return "소형"

    return "알수없음"


def days_since(noticeSdt: Optional[str]) -> int:
    if not noticeSdt:
        return 0
    try:
        dt = datetime.strptime(noticeSdt, "%Y%m%d")
        return (datetime.now() - dt).days
    except Exception:
        return 0


def _age_in_years(age_field: Optional[str]) -> Optional[float]:
    if not age_field:
        return None
    s = str(age_field)
    # "2025(년생)" -> birth year
    m = re.search(r"(\d{4})", s)
    if m:
        try:
            birth = int(m.group(1))
            return max(0.0, datetime.now().year - birth)
        except Exception:
            pass
    # "2살" or "3" 등
    m2 = re.search(r"(\d+(\.\d+)?)", s)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return None


def _text_has_keywords(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for k in keywords:
        if k.lower() in t:
            return True
    return False


def _map_activity_level(level: str) -> float:
    return {"매우 활발함": 0.9, "보통": 0.6, "주로 실내 생활": 0.3}.get(level, 0.6)


def _map_care_time(cat: str) -> float:
    return {"10분 이하": 0.2, "30분": 0.4, "1시간": 0.6, "2시간 이상": 0.9}.get(cat, 0.5)


def _daily_home_time_adjustment(daily: str) -> float:
    if not daily:
        return 0.0
    return {"0~4시간": -0.15, "4~8시간": 0.0, "8~12시간": 0.10, "12시간 이상": 0.15}.get(daily, 0.0)


# -------------------------
# 우선순위 점수
# -------------------------

def priority_boost(doc: Dict[str, Any]) -> float:
    """
    장기 보호 / 대형 / 건강 이슈 / 믹스 등에 대한 우선 입양 가점.
    """
    score = 0.0
    # 나이
    age_txt = doc.get("age", "") or ""
    m = re.search(r"(\d{4})", age_txt)
    if m:
        years = datetime.now().year - int(m.group(1))
        if years >= 7:
            score += 1.0
        elif years >= 4:
            score += 0.5
    # 크기
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    if a_size == "대형":
        score += 0.7
    # 건강/특수 -> 건강은 추천에 있어 +- 요소가 둘다 되므로 제외
    # marks = " ".join([
    #     doc.get("specialMark","") or "",
    #     (doc.get("extractedFeature") or {}).get("noticeable_features","") or "",
    #     doc.get("healthChk","") or "",
    # ])
    # if any(w in marks for w in ["불량", "감염", "병", "장애", "실명", "결손", "오염"]):
    #     score += 1.0
    # 장기 체류
    if days_since(doc.get("noticeSdt")) >= 30:
        score += 0.8
    # 믹스
    if doc.get("kindNm") in ["믹스견", "한국 고양이"]:
        score += 0.3
    return score  # 0~3 근사


# -------------------------
# 쿼리 generic한 정도
# -------------------------

def query_domain_specificity(q: str) -> float:
    """
    쿼리가 도메인(유기동물/크기/색/성격 등)에 얼마나 특화되어 있는지 0~1로 반환.
    - 도메인 키워드 hit 개수 (substring + 의미 기반)
    - 토큰 길이(너무 짧은 문장은 generic에 가깝게)
    두 가지를 섞어서 스코어링.
    """
    if not q or len(q.strip()) < 4:
        return 0.0

    ql = q.lower()

    KEYWORDS = [
        # 종 관련
    "개","강아지","dog","멍멍이",
    "고양이","냥이","cat",
    "기타","토끼","햄스터","기니피그","고슴도치","거북이","새","파충류",

    # 크기/털 관련
    "대형","큰 편","size large","라지","big dog","big cat",
    "중형","보통 크기","medium size","middle size",
    "소형","작은 편","small size","little dog","little cat",
    "장모","긴 털","long hair","털이 풍성한",
    "단모","짧은 털","short hair","털이 짧은",
    "중간 길이 털","세미 롱","medium hair",

    # 활동성/성격
    "활발","사교","사람 좋아","장난","에너지","놀이", "활동적",
    "낯가림","겁","경계","소심","예민","조심","민감",
    "병","감염","상처","치료","질환","장애","실명","결손","아픈","염증",
    "피부","털빠짐","비듬","기생충",

    # 성격 태그
    "차분","조용","온순","침착","순함","평온","차분함",
    "활동","신남","뛰","활기","명랑","발랄",
    "독립","자립","혼자 잘","스스로","독립적",
    "애교","교감","스킨십","친밀","붙임성","친화","귀여운",

    # 색/기타 키워드
    "검은색","검정","흰색","하얀","갈색","회색","치즈","베이지","크림", "검은", "흰"
    "점박이","얼룩",
    "아기","짧은 털","긴 털","부드러운","거친","믹스","한국 고양이"
    ,"고양","파충","조류","연한","진한","큰","작은","귀","눈","꼬리","털","무늬", "작"
    ,"건장", "체구 큼","묵직", "large", "big", "아담", "작은 체구", "small", "tiny"
    ]

    # 1) substring 기반 도메인 키워드 hit
    hit_set = {k for k in KEYWORDS if k in ql}
    hit_count = len(hit_set)

    # 2) 의미 기반 도메인 키워드 hit (substring이 거의 없을 때 특히 의미 있음)
    try:
        q_emb = _cached_text_embedding(ql)
        THRESHOLD = 0.60
        for k in KEYWORDS:
            if k in hit_set:
                continue
            kw_emb = _get_keyword_embedding(k)
            sim = cosine_similarity(q_emb, kw_emb)
            if sim >= THRESHOLD:
                hit_set.add(k)
    except Exception:
        pass

    hit_count = len(hit_set)

    # 3) 쿼리 길이(토큰 수) - 너무 짧으면 generic한 편
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", ql)
    token_count = len(tokens)

    # 최대 4개 도메인 키워드를 기준으로 0~1 스케일
    domain_factor = min(hit_count / 4.0, 1.0)
    # 토큰 6개 이상이면 길이 factor 1.0
    length_factor = min(token_count / 6.0, 1.0)

    # 도메인 키워드 히트에 좀 더 가중치(0.7), 길이(0.3)
    spec = 0.7 * domain_factor + 0.3 * length_factor
    return _clamp01(spec)

# -------------------------
# 설문 → 프로필 텍스트
# -------------------------

def build_profile_text(ans: Dict[str, Any]) -> str:
    if not ans:
        return ""
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
    """
    현재는 하드 필터 없음.
    (알레르기는 스코어링 단계에서 강한 감점으로 처리)
    """
    return True


def size_match(pref: str, a_size: str) -> float:
    if pref in ("", "상관없음") or a_size == "알수없음":
        return 0.2
    if pref == a_size:
        return 1.0
    if (pref, a_size) in [("소형","중형"),("중형","소형"),("대형","중형"),("중형","대형")]:
        return 0.5
    return 0.0


# -------------------------
# 활동성 / 케어 추정
# -------------------------

def apply_breed_activity(base_activity: float, upkind: str, kind: str) -> float:
    kind = kind.replace(" ", "").lower()

    # 고양이 활동성 테이블
    CAT_ACTIVITY_SCORE = {
        "벵갈": 0.25, "아비시니안": 0.20, "시암": 0.18, "샴": 0.18,
        "싱가퓨라": 0.20, "소말리": 0.20,
        "러시안블루": 0.05, "브리티시쇼트헤어": 0.00, "한국고양이": 0.05,
        "메인쿤": 0.05, "노르웨이숲": 0.05, "터키시앙고라": 0.10,
        "페르시안": -0.15, "스코티시폴드": -0.10, "렉돌": -0.05,
    }

    # 개 활동성 테이블
    DOG_ACTIVITY_SCORE = {
        "보더콜리": 0.35, "저먼셰퍼드독": 0.30,
        "벨지안셰퍼드독": 0.30, "래브라도리트리버": 0.25,
        "골든리트리버": 0.25, "사모예드": 0.25,
        "시베리안허스키": 0.30, "말라뮤트": 0.25,
        "진도견": 0.25, "풍산견": 0.25,
        "잭러셀테리어": 0.30, "비글": 0.25,
        "시바": 0.10, "웰시코기": 0.10, "셔틀랜드쉽독": 0.15,
        "슈나우저": 0.10, "토이푸들": 0.05, "포메라니안": 0.05,
        "불독": -0.20, "프렌치불독": -0.20, "퍼그": -0.20,
        "페키니즈": -0.15, "시츄": -0.10, "차우차우": -0.15,
        "그레이트피레니즈": -0.05,
    }

    score_map = CAT_ACTIVITY_SCORE if upkind == "고양이" else DOG_ACTIVITY_SCORE if upkind == "개" else {}

    for breed, score in score_map.items():
        if breed in kind:
            return base_activity + score

    return base_activity

def infer_activity_and_care(doc: Dict[str, Any]) -> Tuple[float, float]:
    base_activity = 0.50
    grooming_base = 0.45
    medical_need = 0.0

    upkind = (doc.get("upKindNm") or "").strip().lower()
    kindNm = (doc.get("kindNm") or "").lower()

    base_activity = apply_breed_activity(base_activity, upkind, kindNm)

    SIZE_LARGE_WORDS = ["대형", "큰", "건장", "체구 큼", "묵직", "large", "big"]
    SIZE_SMALL_WORDS = ["소형", "작", "아담", "작은 체구", "small", "tiny"]

    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    rsize = a_size.lower()
    if (
        _text_has_keywords(rsize, SIZE_LARGE_WORDS)
        or _semantic_has_keywords(rsize, SIZE_LARGE_WORDS, threshold=0.55)
    ):
        base_activity += 0.05

    if (
        _text_has_keywords(rsize, SIZE_SMALL_WORDS)
        or _semantic_has_keywords(rsize, SIZE_SMALL_WORDS, threshold=0.55)
    ):
        base_activity -= 0.05

    age_yrs = _age_in_years(doc.get("age") or "")
    if age_yrs is not None:
        if age_yrs <= 2.0:
            base_activity += 0.10
        elif age_yrs >= 7.0:
            base_activity -= 0.20

    text_fields = " ".join([
        (doc.get("specialMark") or ""),
        (doc.get("extractedFeature") or {}).get("health_impression", "") or "",
        (doc.get("extractedFeature") or {}).get("noticeable_features", "") or ""
    ]).lower()

    POS_ACTIVITY_WORDS = ["활발", "사교", "사람 좋아", "장난", "에너지", "놀이"]
    NEG_ACTIVITY_WORDS = ["낯가림", "겁", "경계", "소심", "예민", "조심"]
    HEALTH_WORDS       = ["병", "감염", "상처", "치료", "질환", "장애", "실명", "결손", "아픈", "염증"]
    SKIN_COAT_WORDS    = ["피부", "털빠짐", "비듬", "기생충", "염증"]

    # positive activity indicators
    if (
        _text_has_keywords(text_fields, POS_ACTIVITY_WORDS)
        or _semantic_has_keywords(text_fields, POS_ACTIVITY_WORDS, threshold=0.60)
    ):
        base_activity += 0.12

    # negative activity indicators
    if (
        _text_has_keywords(text_fields, NEG_ACTIVITY_WORDS)
        or _semantic_has_keywords(text_fields, NEG_ACTIVITY_WORDS, threshold=0.60)
    ):
        base_activity -= 0.08

    # health-related indicators
    if (
        _text_has_keywords(text_fields, HEALTH_WORDS)
        or _semantic_has_keywords(text_fields, HEALTH_WORDS, threshold=0.60)
    ):
        medical_need = max(medical_need, 0.7)
        base_activity -= 0.25

    neuter = (doc.get("neuterYn") or "").upper()
    if neuter == "Y":
        base_activity -= 0.05

    fur_len = ((doc.get("extractedFeature") or {}).get("fur_length") or "").lower()
    fur_texture = ((doc.get("extractedFeature") or {}).get("fur_texture") or "").lower()

    if "긴" in fur_len or "장모" in fur_len:
        grooming_base = 0.9
    elif "짧" in fur_len:
        grooming_base = 0.2
    elif "중간" in fur_len:
        grooming_base = 0.5
    else:
        # 의미 기반으로 fur_len 추정
        if fur_len.strip():
            THRESHOLD = 0.60
            try:
                f_emb = _cached_text_embedding(fur_len)
                LONG_WORDS   = ["장모", "긴 털", "long hair", "털이 풍성한"]
                SHORT_WORDS  = ["단모", "짧은 털", "short hair", "털이 짧은"]
                MEDIUM_WORDS = ["중간 길이 털", "세미 롱", "medium hair"]

                def any_sim(words: List[str]) -> bool:
                    for w in words:
                        kw_emb = _get_keyword_embedding(w)
                        sim = cosine_similarity(f_emb, kw_emb)
                        if sim >= THRESHOLD:
                            return True
                    return False

                if any_sim(LONG_WORDS):
                    grooming_base = 0.9
                elif any_sim(SHORT_WORDS):
                    grooming_base = 0.2
                elif any_sim(MEDIUM_WORDS):
                    grooming_base = 0.5
                else:
                    grooming_base = 0.45
            except Exception:
                grooming_base = 0.45
        else:
            grooming_base = 0.45

    # skin / coat health issues
    if (
        _text_has_keywords(text_fields, SKIN_COAT_WORDS)
        or _semantic_has_keywords(text_fields, SKIN_COAT_WORDS, threshold=0.60)
    ):
        grooming_base = min(1.0, grooming_base + 0.25)
        medical_need = max(medical_need, 0.6)

    care_score = 0.6 * grooming_base + 0.4 * medical_need
    care_score = _clamp01(care_score)
    activity_score = _clamp01(base_activity)

    activity_score = round(activity_score, 3)
    care_score = round(care_score, 3)

    return (activity_score, care_score)


def compute_desired_activity_and_care(ans: Dict[str, Any]) -> Tuple[float, float]:
    if not ans:
        return 0.6, 0.5

    desired_activity = _map_activity_level(ans.get("activityLevel", ""))
    daily = ans.get("dailyHomeTime", "")
    adj = _daily_home_time_adjustment(daily)
    if adj != 0.0:
        desired_activity = max(0.0, min(1.0, desired_activity + adj))

    try:
        fc = int(ans.get("familyCount", 0))
        if fc >= 3:
            desired_activity = min(1.0, desired_activity + 0.05)
    except Exception:
        pass

    if ans.get("hasChildOrElder") == "있음":
        desired_activity = max(0.0, desired_activity - 0.10)

    ph = ans.get("petHistory", "") or ""
    if ph and ("현재" in ph or "과거" in ph):
        desired_activity = min(1.0, desired_activity + 0.05)

    desired_care = _map_care_time(ans.get("careTime", ""))
    budget = ans.get("budget", "")
    if budget and any(x in budget for x in ["700만원","600만원","500만원"]):
        desired_care = min(1.0, desired_care + 0.05)

    if ans.get("preferredSize") and ans.get("preferredSize") != "":
        if ans.get("preferredSize") == "대형":
            desired_activity = min(1.0, desired_activity + 0.03)

    desired_activity = round(_clamp01(desired_activity), 3)
    desired_care = round(_clamp01(desired_care), 3)

    return desired_activity, desired_care

def compute_user_experience_score(ans: Dict[str, Any]) -> float:
    """
    사용자의 설문 응답을 바탕으로 반려동물 양육 경험 점수를 0-1 사이로 계산합니다.
    """
    if not ans:
        return 0.3  # 정보가 없을 경우 기본값

    pet_history = ans.get("petHistory", "")
    score = 0.0

    if "현재" in pet_history:
        score = 1
    elif "과거" in pet_history:
        score = 0.5
    else: # "없음"
        score = 0.1
    
    # 추가 정보로 미세 조정 (예: 가족 수가 많으면 돌봄에 유리)
    try:
        if int(ans.get("familyCount", 1)) >= 3:
            score = min(1.0, score + 0.05)
    except (ValueError, TypeError):
        pass

    return score

# -------------------------
# compat_score + 디테일
# -------------------------

def compat_score_with_details(ans: Dict[str, Any], doc: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    if not ans:
        return 0.0, {}

    score = 0.0
    wsum = 0.0
    details: Dict[str, Any] = {}

    user_exp = compute_user_experience_score(ans)

    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))

    # 1) 크기
    w_size = 0.25
    sm = size_match(ans.get("preferredSize",""), a_size)
    contrib_size = w_size * sm
    score += contrib_size
    wsum  += w_size
    details["size"] = {
        "weight": w_size,
        "match": sm,
        "contribution": contrib_size,
    }

    # 2) 활동/케어
    desired_active, care_time = compute_desired_activity_and_care(ans)
    a_act, a_care = infer_activity_and_care(doc)
    if doc.get("specialMark") and "활발" in doc.get("specialMark"):
        a_act = max(a_act, 0.8)
    act_match = 1 - min(1.0, abs(desired_active - a_act))
    care_match = 1 - min(1.0, abs(care_time - a_care))
    avg_match = (act_match + care_match) / 2.0

    w_act = 0.35
    contrib_act = w_act * avg_match
    score += contrib_act
    wsum  += w_act
    details["activity_care"] = {
        "weight": w_act,
        "act_match": act_match,
        "care_match": care_match,
        "avg_match": avg_match,
        "contribution": contrib_act,
    }

    # 3) 주거/평수
    small_house = ans.get("houseSize") in ["10평 미만","10평 ~ 20평"]
    apt = ans.get("residenceType") == "아파트"
    house_ok = 1.0
    if (small_house or apt) and a_size == "대형":
        house_ok = 0.1
    elif a_size == "중형" and (small_house or apt):
        house_ok = 0.6

    w_house = 0.2
    contrib_house = w_house * house_ok
    score += contrib_house
    wsum  += w_house
    details["house"] = {
        "weight": w_house,
        "house_ok": house_ok,
        "contribution": contrib_house,
    }

    # 4) 성격
    pref_pers = set(ans.get("preferredPersonality") or [])
    persona_sources = [doc.get("specialMark") or ""]
    persona_text = " ".join(persona_sources).lower()

    TEMPERAMENT_MAP = {
        "차분": ["차분", "조용", "온순", "침착", "순함", "순", "평온", "차분함"],
        "활발": ["활발", "에너지", "신남", "뛰", "활기", "명랑", "발랄"],
        "독립": ["독립", "자립", "혼자 잘", "스스로", "독립적"],
        "애교": ["애교", "교감", "스킨십", "사람 좋아", "친밀", "붙임성", "친화", "귀여운"],
        "예민": ["예민", "민감", "신경질", "경계", "조심스러움", "날카로움"],
    }

    if pref_pers and pref_pers == {"상관없음"}:
        persona_score = 0.5
    else:
        persona_score = 0.2
        if pref_pers:
            hits = 0.0
            for pref in pref_pers:
                if pref not in TEMPERAMENT_MAP and pref != "상관없음":
                    continue
                if pref in TEMPERAMENT_MAP:
                    if any(k in persona_text for k in TEMPERAMENT_MAP[pref]):
                        hits += 1.0
                    else:
                        if pref == "차분" and any(k in persona_text for k in TEMPERAMENT_MAP["예민"]):
                            hits -= 0.3
            effective_pref_count = max(1, len([p for p in pref_pers if p in TEMPERAMENT_MAP]))
            persona_score = min(1.0, hits / effective_pref_count)

    w_pers = 0.2
    contrib_pers = w_pers * persona_score
    score += contrib_pers
    wsum  += w_pers
    details["persona"] = {
        "weight": w_pers,
        "persona_score": persona_score,
        "contribution": contrib_pers,
    }

    if wsum > 0:
        norm_score = score / wsum
        norm_factor = 1.0 / wsum
        for v in details.values():
            v["norm_contribution"] = v["contribution"] * norm_factor
        details["_wsum"] = wsum
    else:
        norm_score = 0.0
        details["_wsum"] = 0.0

    norm_score = _clamp01(norm_score)
    return norm_score, details


# -------------------------
# 의미 기반 성격 매칭
# -------------------------

def find_personality_matches(pref_pers: List[str], doc: Dict[str, Any]) -> List[str]:
    """
    설문에서 사용자가 선택한 선호 성격(pref_pers)과
    동물 문서의 특성 텍스트(특징/특이사항)를
    1) 단순 키워드
    2) 임베딩 코사인 유사도
    두 단계로 매칭해서, 실제로 잘 맞는 성격 라벨 리스트를 리턴.
    """
    raw_text = " ".join([
        doc.get("specialMark", "") or "",
        (doc.get("extractedFeature") or {}).get("noticeable_features", "") or "",
    ]).strip()

    if not raw_text or not pref_pers:
        return []

    persona_text = raw_text.lower()

    TEMPERAMENT_MAP = {
        "차분": ["차분", "조용", "온순", "침착", "순함", "평온"],
        "활발": ["활발", "에너지", "활동", "발랄", "명랑"],
        "독립": ["독립", "자립", "혼자 잘", "스스로"],
        "애교": ["애교", "교감", "사람 좋아", "붙임성", "친화", "귀여운"],
        "예민": ["예민", "민감", "경계"],
    }

    hits: List[str] = []

    for pref in (pref_pers or []):
        if pref not in TEMPERAMENT_MAP:
            continue
        kw_list = TEMPERAMENT_MAP[pref]
        if any(k in persona_text for k in kw_list):
            hits.append(pref)

    if hits:
        return list(sorted(set(hits)))

    try:
        text_emb = _cached_text_embedding(persona_text)
    except Exception:
        return list(sorted(set(hits)))

    THRESHOLD = 0.60

    for pref in (pref_pers or []):
        if pref not in TEMPERAMENT_MAP:
            continue

        for kw in TEMPERAMENT_MAP[pref]:
            try:
                kw_emb = _get_keyword_embedding(kw)
                sim = cosine_similarity(text_emb, kw_emb)
            except Exception:
                continue

            if sim >= THRESHOLD:
                hits.append(pref)
                break

    return list(sorted(set(hits)))

# -------------------------
# 쿼리 키워드 추출 (색/기타)
# -------------------------

def find_query_color_keywords(q: str) -> List[str]:
    if not q:
        return []
    ql = q.lower()
    kws: List[str] = []

    KEYWORDS = [
        "검은색","검정","흰색","하얀","갈색","회색","치즈","베이지","크림",
        "점박이","얼룩"
    ]

    # 1차: substring
    for k in KEYWORDS:
        if k.lower() in ql:
            kws.append(k)

    if kws:
        return kws

    # 2차: 의미 기반
    try:
        q_emb = _cached_text_embedding(ql)
    except Exception:
        return kws

    THRESHOLD = 0.60
    for k in KEYWORDS:
        kw_emb = _get_keyword_embedding(k)
        sim = cosine_similarity(q_emb, kw_emb)
        if sim >= THRESHOLD:
            kws.append(k)

    return kws


def find_query_other_keywords(q: str) -> List[str]:
    if not q:
        return []
    ql = q.lower()
    kws: List[str] = []

    KEYWORDS = [
        "대형","중형","소형","활발","활동","차분","애교","귀여운","아기",
        "짧은 털","긴 털","부드러운","거친","믹스","한국 고양이"
    ]

    # 1차: substring
    for k in KEYWORDS:
        if k.lower() in ql:
            kws.append(k)

    if kws:
        return kws

    # 2차: 의미 기반
    try:
        q_emb = _cached_text_embedding(ql)
    except Exception:
        return kws

    THRESHOLD = 0.60
    for k in KEYWORDS:
        kw_emb = _get_keyword_embedding(k)
        sim = cosine_similarity(q_emb, kw_emb)
        if sim >= THRESHOLD:
            kws.append(k)

    return kws


# -------------------------
# 추가 감점 요인: 알레르기 / 현재 반려동물
# -------------------------

def compute_allergy_penalty(ans: Dict[str, Any], doc: Dict[str, Any]) -> float:
    """
    0~1 사이 값. 1에 가까울수록 강한 감점.
    """
    if not ans or ans.get("hasAllergy") != "있음":
        return 0.0

    aa = (ans.get("allergyAnimal") or "").lower()
    species = (doc.get("upKindNm") or "").lower()

    penalty = 0.0
    if "고양" in aa and "고양" in species:
        penalty = 0.9
    if ("강아지" in aa or "개" in aa or "dog" in aa) and species == "개":
        penalty = max(penalty, 0.9)

    # 기타 동물 알레르리라면 소폭 감점
    if penalty == 0.0 and aa.strip():
        penalty = 0.3

    return _clamp01(penalty)


def compute_pet_conflict_penalty(ans: Dict[str, Any], doc: Dict[str, Any]) -> float:
    """
    현재 반려동물 vs 신규 아이 종 조합에 따른 감점.
    """
    if not ans:
        return 0.0
    current_list = ans.get("currentPets") or []
    current_text = " ".join(current_list).lower()
    if not current_text.strip():
        return 0.0

    species = (doc.get("upKindNm") or "").lower()
    penalty = 0.0

    # 고양이 + 소동물/조류
    if "고양" in species:
        if any(x in current_text for x in ["소동물","햄스터","토끼","기니피그","조류","새","파충류"]):
            penalty = max(penalty, 0.7)

    # 개 + 고양이
    if species == "개":
        if "고양" in current_text:
            penalty = max(penalty, 0.5)

    # 개/고양이 + 조류
    if ("개" in species or "고양" in species) and any(x in current_text for x in ["조류","새"]):
        penalty = max(penalty, 0.5)

    return _clamp01(penalty)


def build_reasons(
    survey: Dict[str, Any],
    q: str,
    meta: Dict[str, float],
    doc: Dict[str, Any]
) -> List[Dict[str, Any]]:
    reasons: List[Dict[str, Any]] = []

    # 1) 임베딩 유사도
    sim = meta.get("sim", 0.0) or 0.0
    if sim > 0.53:
        reasons.append({
            "type": "embedding",
            "label": "검색/프로필 유사도",
            "score": float(sim),
            "evidence": "",
            "reason": "사용자 쿼리·설문 응답과 내용이 잘 맞아요.",
        })

    # 2) 크기 매칭
    pref_size = survey.get("preferredSize") if survey else ""
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size", ""))
    s_size = size_match(pref_size or "", a_size)
    if s_size >= 0.8:
        reasons.append({
            "type": "size",
            "label": "크기 적합도",
            "score": float(s_size),
            "evidence": a_size,
            "reason": f"설문에서 선택한 크기({pref_size})와 실제 크기({a_size})가 잘 맞아요.",
        })
    elif s_size <= 0.0 and pref_size and pref_size != "상관없음":
        reasons.append({
            "type": "size_penalty",
            "label": "크기 부담",
            "score": float(s_size),
            "evidence": a_size,
            "reason": f"선호 크기({pref_size})와 실제 크기({a_size})가 달라서 적합도가 조금 낮을 수 있어요.",
        })

    # 3) 성격 매칭
    pref_pers = (survey.get("preferredPersonality") or []) if survey else []
    pers_hits = find_personality_matches(pref_pers, doc)
    if pers_hits:
        reasons.append({
            "type": "personality",
            "label": "성격 적합도",
            "score": 0.8,
            "evidence": (doc.get("specialMark") or "")[:120],
            "reason": f"설문에서 고른 성격({', '.join(pers_hits)})과 보호소 메모의 성격이 비슷해요.",
        })

    # 4) 활동/케어 적합
    if survey:
        desired_active, care_time = compute_desired_activity_and_care(survey)
        a_act, a_care = infer_activity_and_care(doc)
        act_match = 1 - min(1.0, abs(desired_active - a_act))
        care_match = 1 - min(1.0, abs(care_time - a_care))
        s_act = (act_match + care_match) / 2.0

        if desired_active > 0.7:
            act_reason = "활동적인 반려동물을 선호해요."
        elif desired_active < 0.4:
            act_reason = "차분한 반려동물을 선호해요."
        else:
            act_reason = "보통 수준의 활동성을 선호해요."

        if s_act >= 0.8:
            reasons.append({
                "type": "activity_care",
                "label": "활동성·케어 적합도",
                "score": float(s_act),
                "evidence": (
                    f"동물 활동성 추정: {a_act}, 사용자 기대: {desired_active}, "
                    f"케어 요구: {a_care}, 사용자 케어 가능 시간: {care_time}"
                ),
                "reason": act_reason + " 동물의 추정 활동·케어 요구와 잘 맞아요.",
            })
        elif s_act <= 0.3:
            reasons.append({
                "type": "activity_care_penalty",
                "label": "활동성·케어 차이",
                "score": float(s_act),
                "evidence": (
                    f"동물 활동성 추정: {a_act}, 사용자 기대: {desired_active}, "
                    f"케어 요구: {a_care}, 사용자 케어 가능 시간: {care_time}"
                ),
                "reason": "활동량이나 케어 난이도가 현재 생활 패턴과 차이가 커서 부담이 될 수 있어요.",
            })

    # 5) 위치 관련 (가·감점 모두)
    user_addr = survey.get("address") if survey else None
    loc_score = 0.0
    try:
        if user_addr:
            loc_score = location_score(user_addr, doc.get("careAddr"))
            if loc_score is None:
                loc_score = 0.0
    except Exception:
        loc_score = 0.0

    if loc_score > 0.7:
        reasons.append({
            "type": "location",
            "label": "지역 근접성",
            "score": float(loc_score),
            "evidence": doc.get("careAddr") or "",
            "reason": "현재 거주지와 보호소 거리가 가까워 방문·입양 절차가 비교적 편리해요.",
        })
    elif 0.0 < loc_score < 0.4:
        reasons.append({
            "type": "location_penalty",
            "label": "거리 부담",
            "score": float(loc_score),
            "evidence": doc.get("careAddr") or "",
            "reason": "보호소와 거리가 멀어서 방문·이동에 시간이 더 들 수 있어요.",
        })

    # 6) 우선순위(장기 보호 / 건강 등)
    prio = meta.get("prio", 0.0) or 0.0
    if prio > 1.5:
        reasons.append({
            "type": "priority",
            "label": "우선 입양 대상",
            "score": float(prio),
            "evidence": doc.get("specialMark") or "",
            "reason": "나이·건강·장기 보호 등 이유로 특히 입양이 시급한 아이예요.",
        })

    # 7) 키워드 매칭 근거
    matched_kw: List[str] = []
    try:
        q_c_keywords = find_query_color_keywords(q)
        text_for_search_color = " ".join([
            doc.get("colorCd", "") or "",
            doc.get("main_color", "") or "",
        ]).lower()
        for kw in q_c_keywords:
            if kw.lower() in text_for_search_color:
                matched_kw.append(kw)

        q_o_keywords = find_query_other_keywords(q)
        text_for_search_other = " ".join([
            doc.get("kindNm", "") or "",
            doc.get("specialMark", "") or "",
            (doc.get("extractedFeature") or {}).get("noticeable_features", "") or "",
            (doc.get("extractedFeature") or {}).get("fur_length", "") or "",
            (doc.get("extractedFeature") or {}).get("fur_texture", "") or "",
            (doc.get("extractedFeature") or {}).get("fur_pattern", "") or "",
        ]).lower()
        for kw in q_o_keywords:
            if kw.lower() in text_for_search_other:
                matched_kw.append(kw)
    except Exception:
        matched_kw = []

    if matched_kw:
        uniq_kw = sorted(set(matched_kw))
        reasons.append({
            "type": "keyword",
            "label": "키워드 매칭",
            "score": 0.5,
            "evidence": ", ".join(uniq_kw),
            "reason": f"입력한 키워드({', '.join(uniq_kw)})와 색·특징이 잘 맞아요.",
        })

    # -----------------------------
    # 여기부터 감점 중심 디테일
    # -----------------------------
    comp_details = meta.get("comp_details") or {}

    # (a-1) 집/평수 vs 크기 미스매치
    house_det = comp_details.get("house")
    if house_det:
        house_ok = float(house_det.get("house_ok", 1.0))
        norm_contrib = float(house_det.get("norm_contribution", 0.0))
        if house_ok <= 0.3 and norm_contrib < -0.15:
            reasons.append({
                "type": "penalty_house",
                "label": "공간 제약",
                "score": norm_contrib,
                "evidence": f"house_ok={round(house_ok,2)}",
                "reason": "주거 공간 대비 크기 부담이 있을 수 있어요.",
            })

        # (a-2) 집이 넉넉해서 가점
        if house_ok >= 0.9 and norm_contrib > 0.05:
            reasons.append({
                "type": "house_positive",
                "label": "공간 여유",
                "score": norm_contrib,
                "evidence": f"house_ok={round(house_ok,2)}",
                "reason": "생활 공간이 충분해 크기 면에서 여유로운 친구예요.",
            })

    # (b) 활동성/케어 mismatch
    act_det = comp_details.get("activity_care")
    if act_det:
        avg_match = float(act_det.get("avg_match", 1.0))
        norm_contrib = float(act_det.get("norm_contribution", 0.0))
        if avg_match <= 0.4 and norm_contrib < -0.15:
            reasons.append({
                "type": "penalty_activity_care",
                "label": "활동/케어 부담",
                "score": norm_contrib,
                "evidence": f"avg_match={round(avg_match,2)}",
                "reason": "활동량이나 케어 난이도가 현재 생활 패턴보다 높을 수 있어요.",
            })

    # (c) 알레르기 감점
    allergy_penalty = float(meta.get("allergy_penalty", 0.0))
    if allergy_penalty >= 0.5:
        reasons.append({
            "type": "penalty_allergy",
            "label": "알레르기 위험",
            "score": -allergy_penalty,
            "evidence": survey.get("allergyAnimal") if survey else "",
            "reason": "알레르기 위험이 커서 생활에 불편을 줄 수 있어요.",
        })
    elif allergy_penalty >= 0.2:
        reasons.append({
            "type": "penalty_allergy_light",
            "label": "알레르기 가능성",
            "score": -allergy_penalty,
            "evidence": survey.get("allergyAnimal") if survey else "",
            "reason": "알레르기 가능성이 있어 주의가 필요해요.",
        })

    # (d) 현재 반려동물과의 궁합 감점
    conflict_penalty = float(meta.get("pet_conflict_penalty", 0.0))
    if conflict_penalty >= 0.4:
        reasons.append({
            "type": "penalty_pet_conflict",
            "label": "현재 반려동물과 궁합",
            "score": -conflict_penalty,
            "evidence": ", ".join(survey.get("currentPets", [])) if survey else "",
            "reason": "현재 반려동물과 종/습성 상 충돌 가능성이 있어요.",
        })

    # (e) 위치가 많이 먼 경우 감점 (상세용)
    loc_component = float(meta.get("loc_component", 0.0))
    if loc_score <= 0.3 and loc_component < 0:
        reasons.append({
            "type": "penalty_location",
            "label": "거리 부담",
            "score": loc_component,
            "evidence": doc.get("careAddr") or "",
            "reason": "보호소가 거주지에서 꽤 먼 편이라 이동이 부담될 수 있어요.",
        })

    return reasons

# -------------------------
# 요청 모델
# -------------------------

class RecommendRequest(BaseModel):
    natural_query: str
    limit: int = 3


class HybridRequest(BaseModel):
    natural_query: str
    limit: int = 6
    user_id: Optional[str] = None
    use_survey: bool = True
    # 프론트에서 즐겨찾기 desertionNo 리스트를 넘겨 줄 것
    favorite_desertion_nos: Optional[List[str]] = None


# -------------------------
# 단순 recommend (기존)
# -------------------------

@router.post("")
def recommend_animals(body: RecommendRequest):
    try:
        user_emb = get_embedding(body.natural_query)
    except Exception as e:
        raise HTTPException(500, f"임베딩 생성 오류: {e}")

    species_list = extract_species(body.natural_query)
    query_filter: Dict[str, Any] = {}
    if species_list:
        if len(species_list) == 1:
            query_filter = {"upKindNm": species_list[0]}
        else:
            query_filter = {"upKindNm": {"$in": species_list}}

    candidates: List[Tuple[float, Dict[str,Any]]] = []
    for doc in collection.find(query_filter):
        animal_emb = np.array(doc.get("embedding", []), dtype=np.float32)
        if animal_emb.size == 0:
            continue
        score = cosine_similarity(user_emb, animal_emb)
        candidates.append((score, doc))

    topn = sorted(candidates, key=lambda x: x[0], reverse=True)[:body.limit]
    return [
        {
            "score": round(s, 4),
            "desertionNo": doc.get("desertionNo"),
            "kindFullNm": doc.get("kindFullNm"),
        }
        for (s, doc) in topn if s > 0.001
    ]


# -------------------------
# 하이브리드 추천
# -------------------------

@router.post("/hybrid")
def recommend_hybrid(body: HybridRequest):
    # 1) 쿼리/프로필 임베딩
    q = (body.natural_query or "").strip()
    try:
        q_emb = _cached_text_embedding(q) if q else None
    except Exception as e:
        raise HTTPException(500, f"임베딩 생성 오류: {e}")

    survey: Dict[str, Any] = {}
    if body.use_survey and body.user_id:
        sdoc = survey_col.find_one({"userId": body.user_id}, sort=[("_id",-1)])
        if sdoc:
            survey = sdoc.get("answers") or sdoc

    profile_text = build_profile_text(survey) if survey else ""
    p_emb = _cached_text_embedding(profile_text) if profile_text else None

    # 쿼리의 도메인 특이성(0~1): 도메인 키워드 hit 개수 + 쿼리 길이 기반
    specificity = query_domain_specificity(q)

    # alpha: 쿼리 임베딩 vs 프로필 임베딩 비율
    #  - generic(0)에 가까우면 0.0
    #  - 매우 도메인 특화(1)에 가까우면 0.7

    # 필드별 임베딩 (색상, 활동성, 외모, 성격)
    try:
        user_fields = user_to_4texts(q, survey)  # returns dict with keys: color, activity, appearance, personality
        # order them to create batch
        field_order = ["color", "activity", "appearance", "personality"]
        user_texts = [user_fields.get(k, "") for k in field_order]
        user_vecs = get_embeddings_batch(user_texts)  # list of np arrays corresponding to user_texts
        user_field_embs = {k: v for k, v in zip(field_order, user_vecs)}
    except Exception as e:
        # if embedding fails, fallback to None for all
        user_field_embs = {k: None for k in ["color","activity","appearance","personality"]}

    # tuning parameter: how much weight to give field-level matches vs global sim_mix

    # 감점 가중치 (알레르기/현 반려동물 충돌은 강한 페널티)

    # 2) 종 필터: 설문 선호 종 + 쿼리 종
    species_from_survey = survey_preferred_species(survey) if survey else []
    species_from_query  = extract_species(q)

    base_species: List[str] = []
    if species_from_survey and species_from_query:
        inter = sorted(set(species_from_survey) & set(species_from_query))
        base_species = inter if inter else sorted(set(species_from_survey) | set(species_from_query))
    elif species_from_survey:
        base_species = species_from_survey
    else:
        base_species = species_from_query

    base_filter: Dict[str, Any] = {}
    if base_species:
        if len(base_species) == 1:
            base_filter["upKindNm"] = base_species[0]
        else:
            base_filter["upKindNm"] = {"$in": base_species}

    # 3) 즐겨찾기 동물 임베딩 준비
    favorite_docs: List[Dict[str, Any]] = []
    valid_fav_embs_by_species: Dict[str, List[np.ndarray]] = {}
    if body.favorite_desertion_nos:
        fav_cursor = collection.find({"desertionNo": {"$in": body.favorite_desertion_nos}})
        favorite_docs = list(fav_cursor)

        if q_emb is not None:
            FAV_Q_THRESHOLD = 0.30
            for fdoc in favorite_docs:
                f_emb = np.array(fdoc.get("embedding") or [], dtype=np.float32)
                if f_emb.size == 0:
                    continue
                fq_sim = cosine_similarity(q_emb, f_emb)
                if fq_sim < FAV_Q_THRESHOLD:
                    # 쿼리와 너무 안 맞는 즐겨찾기는 필터에 영향 주지 않음
                    continue
                species = (fdoc.get("upKindNm") or "").strip()
                if not species:
                    continue
                valid_fav_embs_by_species.setdefault(species, []).append(f_emb)

    # 4) 후보 생성 + 유사도 계산
    candidates: List[Tuple[float, Dict[str,Any], Dict[str,Any], Dict[str, Any]]] = []

    w_sim, w_comp, w_prio, w_loc = 0.40, 0.50, 0.05, 0.05
    w_allergy, w_conflict = 0.8, 0.5
    alpha = 0.7 * specificity  # 0.0 ~ 0.7
    w_sim, w_comp = 0.35+0.1*specificity, 0.55-0.1*specificity

    for doc in collection.find(base_filter):
        is_cat = (doc.get("upKindNm") == "고양이")
        is_dog = (doc.get("upKindNm") == "개")
        age_years = _age_in_years(doc.get("age"))
        is_puppy_kitten = (age_years is not None and age_years <= 1.0)
        if is_cat:
            w_sim *= 1.1
        elif is_dog:
            w_comp *= 1.1
        if is_puppy_kitten:
            w_sim *= 1.15
        else:
            w_comp *= 1.15

        a_emb = np.array(doc.get("embedding") or [], dtype=np.float32)
        if a_emb.size == 0:
            continue

        sim_q = cosine_similarity(q_emb, a_emb) if q_emb is not None else 0.0
        sim_p = cosine_similarity(p_emb, a_emb) if p_emb is not None else 0.0
        base_mix = (alpha * sim_q) + ((1 - alpha) * sim_p) if (q_emb is not None or p_emb is not None) else 0.0

        field_scores: Dict[str, float] = {"color":0.0, "activity":0.0, "appearance":0.0, "personality":0.0}
        try:
            doc_field_embs_raw = doc.get("fieldEmbeddings") or {}
            for k in ["color","activity","appearance","personality"]:
                arr = doc_field_embs_raw.get(k)
                if arr:
                    fvec = np.array(arr, dtype=np.float32)
                else:
                    fvec = np.zeros((1536,), dtype=np.float32)
                uvec = user_field_embs.get(k)
                if uvec is None or uvec.size == 0:
                    fsim = 0.0
                else:
                    fsim = cosine_similarity(uvec, fvec)
                field_scores[k] = float(fsim)
        except Exception:
            # on any error, treat field scores as zeroes
            field_scores = {k:0.0 for k in field_scores.keys()}
        field_weights = {"color":0.35, "activity":0.35, "appearance":0.20, "personality":0.10}
        total_w = sum(field_weights.values())
        field_match_score = sum(field_scores[k] * (field_weights[k] / total_w) for k in field_scores.keys())

        # 즐겨찾기 기반 유사도 (같은 종만)
        sim_f = 0.0
        species = (doc.get("upKindNm") or "").strip()
        fav_emb_list = valid_fav_embs_by_species.get(species) or []
        if fav_emb_list:
            sim_f = max(cosine_similarity(a_emb, fe) for fe in fav_emb_list)

        sim_mix = merge_similarity_scores(base_mix, sim_f, field_match_score)

        if sim_mix <= 0:  # 완전 무관한 경우 제거
            continue

        candidates.append((sim_mix, doc, {"sim_q": sim_q, "sim_p": sim_p, "sim_f": sim_f, "field_match": field_scores}))

    if not candidates:
        return []

    # 유사도 기준 필터: 평균 이상 + 상위 30마리 이내
    sims_only = [c[0] for c in candidates]
    mean_sim = sum(sims_only) / len(sims_only)
    filtered_by_sim = [c for c in candidates if c[0] >= mean_sim]
    filtered_by_sim.sort(key=lambda x: x[0], reverse=True)
    pool = filtered_by_sim[:30]  # 상위 30마리

    if not pool:
        pool = sorted(candidates, key=lambda x: x[0], reverse=True)[:30]

    # 5) 설문 제약(현재는 하드 필터 없음, 구조만 유지)
    filtered: List[Tuple[float, Dict[str,Any], Dict[str,Any]]] = []
    for sim_mix, doc, extra_meta in pool:
        if survey and not survey_constraints(survey, doc):
            continue
        filtered.append((sim_mix, doc, extra_meta))
    if not filtered:
        filtered = pool

    # 6) 최종 스코어링: compat + priority + location ± penalty
    scored: List[Tuple[float, Dict[str,Any], Dict[str,Any]]] = []
    user_addr = survey.get("address") if survey else None

    for sim_mix, doc, extra_meta in filtered:
        comp, comp_details = compat_score_with_details(survey, doc) if survey else (0.0, {})
        prio = priority_boost(doc)
        loc_raw = location_score(user_addr, doc.get("careAddr")) if user_addr else 0.0
        loc_component = (loc_raw - 0.5) * 2.0

        allergy_penalty  = compute_allergy_penalty(survey, doc)
        conflict_penalty = compute_pet_conflict_penalty(survey, doc)

        sim_term      = w_sim * sim_mix
        comp_term     = w_comp * comp
        prio_term     = w_prio * (prio / 3.0)
        loc_term      = w_loc * loc_component
        allergy_term  = -w_allergy * allergy_penalty
        conflict_term = -w_conflict * conflict_penalty

        final_raw = sim_term + comp_term + prio_term + loc_term + allergy_term + conflict_term
        final = _clamp01(final_raw)

        meta = {
            "sim": sim_mix,
            "sim_q": extra_meta.get("sim_q", 0.0),
            "sim_p": extra_meta.get("sim_p", 0.0),
            "sim_f": extra_meta.get("sim_f", 0.0),
            "comp": comp,
            "prio": prio,
            "loc": loc_raw,
            "field_match": extra_meta.get("field_match", {}),
            "loc_component": loc_component,
            "allergy_penalty": allergy_penalty,
            "pet_conflict_penalty": conflict_penalty,
            "comp_details": comp_details,
            "term_contrib": {
                "sim": sim_term,
                "comp": comp_term,
                "prio": prio_term,
                "loc": loc_term,
                "allergy": allergy_term,
                "conflict": conflict_term,
            },
        }
        scored.append((final, doc, meta))


    scored.sort(key=lambda x: x[0], reverse=True)
    topn = scored[: body.limit]

    results = []
    for (s, d, meta) in topn:
        reasons = build_reasons(
            survey,
            q,
            meta,
            d,
        )

        results.append({
            "final": safe_round(s, 4, default=0.0),
            "sim": safe_round(meta.get("sim", 0.0), 4, default=0.0),
            "compat": safe_round(meta.get("comp", 0.0), 4, default=0.0),
            "priority": safe_round(meta.get("prio", 0.0), 3, default=0.0),
            "location": safe_round(meta.get("loc", 0.0), 4, default=0.0),
            "field_match": meta.get("field_match", {}),
            "desertionNo": d.get("desertionNo"),
            "kindFullNm": d.get("kindFullNm"),
            "upKindNm": d.get("upKindNm"),
            "colorCd": d.get("colorCd"),
            "age": d.get("age"),
            "careAddr": d.get("careAddr"),
            "specialMark": d.get("specialMark"),
            "extractedFeature": d.get("extractedFeature"),
            "reasons": reasons,
        })
    return results


# @router.post("/hybrid")
# def recommend_hybrid(body: HybridRequest):
#     # --- (변경 없음) 1단계-A: 기본 정보 및 임베딩 준비 ---
#     q = (body.natural_query or "").strip()
#     try:
#         q_emb = _cached_text_embedding(q) if q else None
#     except Exception as e:
#         raise HTTPException(500, f"임베딩 생성 오류: {e}")

#     survey: Dict[str, Any] = {}
#     if body.use_survey and body.user_id:
#         sdoc = survey_col.find_one({"userId": body.user_id}, sort=[("_id",-1)])
#         if sdoc:
#             survey = sdoc.get("answers") or sdoc

#     if not survey:
#          raise HTTPException(404, f"유저 ID '{body.user_id}'에 대한 설문 데이터를 찾을 수 없습니다.")

#     profile_text = build_profile_text(survey)
#     p_emb = _cached_text_embedding(profile_text) if profile_text else None
#     alpha = 0.3 if is_generic_query(q) else 0.7

#     # --- (변경) 1단계-B: 후보군 생성 (Recall) ---
#     # 목표: 최대한 가볍게, 가능성 있는 후보 200개를 빠르게 추린다.
#     # 복잡한 계산은 모두 제외하고 'sim_mix' 점수만 사용한다.
#     species_from_query = extract_species(q)
#     base_species: List[str] = []
#     if species_from_query:
#         base_species = species_from_query
#     elif survey:
#         species_from_survey = survey_preferred_species(survey)
#         if species_from_survey:
#             base_species = species_from_survey

#     # 최종 DB 필터 생성
#     base_filter: Dict[str, Any] = {}
#     if base_species:
#         if len(base_species) == 1:
#             base_filter["upKindNm"] = base_species[0]
#         else:
#             base_filter["upKindNm"] = {"$in": base_species}

#     initial_candidates = []
#     for doc in collection.find(base_filter):
#         a_emb = np.array(doc.get("embedding", []), dtype=np.float32)
#         if a_emb.size == 0:
#             continue
        
#         sim_q = cosine_similarity(q_emb, a_emb) if q_emb is not None else 0.0
#         sim_p = cosine_similarity(p_emb, a_emb) if p_emb is not None else 0.0
#         sim_mix = (alpha * sim_q) + ((1 - alpha) * sim_p)
        
#         # sim_mix 점수와 doc만 저장. 이 단계에서 다른 계산은 하지 않음.
#         initial_candidates.append((sim_mix, doc))

#     # sim_mix 점수가 높은 순으로 정렬하여 상위 50개만 pool로 사용
#     initial_candidates.sort(key=lambda x: x[0], reverse=True)
#     candidate_pool = initial_candidates[:50]
    
#     if not candidate_pool:
#         return []

#     # --- (변경) 2단계: LTR 모델로 재정렬 (Re-ranking) ---
#     # 목표: 1단계에서 추려진 200개 후보에 대해서만 정밀한 점수를 계산하고 LTR 모델로 최종 순위를 매긴다.

#     if LTR_MODEL is None:
#         raise HTTPException(503, "추천 모델을 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.")

#     # 재정렬할 후보들의 피처와 메타 정보를 담을 리스트
#     to_rank_items = []
#     for sim_mix, doc in candidate_pool: # pool에는 (sim_mix, doc) 튜플이 들어있음
#         # 이 단계에서 비로소 무거운 점수들을 계산
#         comp, comp_details = compat_score_with_details(survey, doc)
#         prio = priority_boost(doc)
#         loc = location_score(survey.get("address"), doc.get("careAddr")) if survey.get("address") else 0.0
#         allergy = compute_allergy_penalty(survey, doc)
#         conflict = compute_pet_conflict_penalty(survey, doc)

#         # LTR 피처 생성 및 이유 생성에 필요한 모든 정보를 meta에 저장
#         meta = {
#             "sim": sim_mix,
#             "comp": comp,
#             "prio": prio,
#             "loc": loc,
#             "allergy_penalty": allergy,
#             "pet_conflict_penalty": conflict,
#             "comp_details": comp_details  # 이유 생성 시 필요
#         }
        
#         # 학습 때와 동일한 함수를 사용해 피처 벡터 생성
#         features = extract_features_for_ltr(doc, meta)
        
#         to_rank_items.append({'doc': doc, 'meta': meta, 'features': features})

#     # 모든 후보의 피처 벡터를 한 번에 모델에 전달하여 랭킹 점수 예측
#     feature_vectors = [item['features'] for item in to_rank_items]
#     ranking_scores = LTR_MODEL.predict(feature_vectors)

#     # 예측된 점수를 각 후보와 다시 매핑
#     scored_candidates = []
#     for i, item in enumerate(to_rank_items):
#         scored_candidates.append((ranking_scores[i], item['doc'], item['meta']))
    
#     # LTR 모델이 예측한 점수가 높은 순으로 최종 정렬
#     scored_candidates.sort(key=lambda x: x[0], reverse=True)


#     # --- (변경 없음) 3단계: 결과 포맷팅 및 반환 ---
#     top_n = scored_candidates[:body.limit]
#     results = []
#     for score, doc, meta in top_n:
#         results.append({
#             "final_score": safe_round(score, 4), # LTR 모델의 예측 점수
#             "sim": safe_round(meta.get("sim", 0.0), 4),
#             "compat": safe_round(meta.get("comp", 0.0), 4),
#             "desertionNo": doc.get("desertionNo"),
#             "kindFullNm": doc.get("kindFullNm"),
#             # meta 데이터가 이미 계산되어 있으므로 build_reasons에 바로 전달
#             "reasons": build_reasons(survey, q, meta, doc),
#             "upKindNm": doc.get("upKindNm"),
#             "colorCd": doc.get("colorCd"),
#             "age": doc.get("age"),
#             "careAddr": doc.get("careAddr"),
#             "specialMark": doc.get("specialMark"),
#         })
        
#     return results