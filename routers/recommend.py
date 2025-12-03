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
from utils.filed_embedding import user_to_4texts

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
fav_col = db["favorites"]

app = FastAPI()

DOG_BREEDS = [
  '모든 품종',
  '믹스견',
  '골든 리트리버',
  '그레이 하운드',
  '그레이트 덴',
  '그레이트 피레니즈',
  '기타',
  '꼬똥 드 뚤레아',
  '네오폴리탄 마스티프',
  '노르포크 테리어',
  '노리치 테리어',
  '노퍽 테리어',
  '뉴펀들랜드',
  '닥스훈트',
  '달마시안',
  '댄디 딘몬트 테리어',
  '도고 까니리오',
  '도고 아르젠티노',
  '도베르만',
  '도사',
  '도사 믹스견',
  '동경견',
  '라브라도 리트리버',
  '라사 압소',
  '라이카',
  '래빗 닥스훈드',
  '랫 테리어',
  '레이크랜드 테리어',
  '로디지안 리즈백',
  '로트와일러',
  '로트와일러 믹스견',
  '마리노이즈',
  '마스티프',
  '말라뮤트',
  '말티즈',
  '맨체스터테리어',
  '미니어쳐 닥스훈트',
  '미니어쳐 불 테리어',
  '미니어쳐 슈나우저',
  '미니어쳐 푸들',
  '미니어쳐 핀셔',
  '미디엄 푸들',
  '미텔 스피츠',
  '바센지',
  '바셋 하운드',
  '버니즈 마운틴 독',
  '베들링턴 테리어',
  '벨기에 그로넨달',
  '벨기에 쉽독',
  '벨기에 테뷰런',
  '벨지안 셰퍼드 독',
  '보더 콜리',
  '보르조이',
  '보스턴 테리어',
  '복서',
  '볼로네즈',
  '부비에 데 플랑드르',
  '불 테리어',
  '불독',
  '브뤼셀그리펀',
  '브리타니 스파니엘',
  '블랙 테리어',
  '비글',
  '비숑 프리제',
  '비어디드 콜리',
  '비즐라',
  '빠삐용',
  '사모예드',
  '살루키',
  '삽살개',
  '샤페이',
  '세인트 버나드',
  '센트럴 아시안 오브차카',
  '셔틀랜드 쉽독',
  '셰퍼드',
  '슈나우져',
  '스코티쉬 테리어',
  '스코티시 디어하운드',
  '스태퍼드셔 불 테리어',
  '스태퍼드셔 불 테리어 믹스견',
  '스탠다드 푸들',
  '스피츠',
  '시바',
  '시베리안 허스키',
  '시베리안라이카',
  '시잉프랑세즈',
  '시츄',
  '시코쿠',
  '실리햄 테리어',
  '실키테리어',
  '아나톨리안 셰퍼드',
  '아메리칸 불독',
  '아메리칸 스태퍼드셔 테리어',
  '아메리칸 스태퍼드셔 테리어 믹스견',
  '아메리칸 아키다',
  '아메리칸 에스키모',
  '아메리칸 코카 스파니엘',
  '아메리칸 핏불 테리어',
  '아메리칸 핏불 테리어 믹스견',
  '아메리칸불리',
  '아이리쉬 레드 앤 화이트 세터',
  '아이리쉬 세터',
  '아이리쉬 울프 하운드',
  '아이리쉬소프트코튼휘튼테리어',
  '아키다',
  '아프간 하운드',
  '알라스칸 말라뮤트',
  '에어델 테리어',
  '오브차카',
  '오스트랄리안 셰퍼드 독',
  '오스트랄리안 캐틀 독',
  '오스트레일리안 켈피',
  '올드 잉글리쉬 불독',
  '올드 잉글리쉬 쉽독',
  '와이마라너',
  '와이어 폭스 테리어',
  '요크셔 테리어',
  '울프독',
  '웨스트 시베리언 라이카',
  '웨스트하이랜드화이트테리어',
  '웰시 코기 카디건',
  '웰시 코기 펨브로크',
  '웰시 테리어',
  '이탈리안 그레이 하운드',
  '잉글리쉬 세터',
  '잉글리쉬 스프링거 스파니엘',
  '잉글리쉬 코카 스파니엘',
  '잉글리쉬 포인터',
  '자이언트 슈나우져',
  '재패니즈 스피츠',
  '잭 러셀 테리어',
  '저먼 셰퍼드 독',
  '저먼 와이어헤어드 포인터',
  '저먼 포인터',
  '저먼 헌팅 테리어',
  '제주개',
  '제페니즈칭',
  '진도견',
  '차우차우',
  '차이니즈 크레스티드 독',
  '치와와',
  '카네 코르소',
  '카레리안 베어독',
  '카이훗',
  '캐벌리어 킹 찰스 스파니엘',
  '케니스펜더',
  '케리 블루 테리어',
  '케언 테리어',
  '케인 코르소',
  '코리아 트라이 하운드',
  '코리안 마스티프',
  '코카 스파니엘',
  '코카 푸',
  '코카시안오브차카',
  '콜리',
  '클라인스피츠',
  '키슈',
  '키스 훈드',
  '토이 맨체스터 테리어',
  '토이 푸들',
  '티베탄 마스티프',
  '파라오 하운드',
  '파슨 러셀 테리어',
  '팔렌',
  '퍼그',
  '페키니즈',
  '페터데일테리어',
  '포메라니안',
  '포인터',
  '폭스테리어',
  '푸들',
  '풀리',
  '풍산견',
  '프레사까나리오',
  '프렌치 불독',
  '프렌치 브리타니',
  '플랫 코티드 리트리버',
  '플롯하운드',
  '피레니안 마운틴 독',
  '필라 브라질레이로',
  '핏불테리어',
  '핏불테리어 믹스견',
  '허배너스',
  '화이트리트리버',
  '화이트테리어',
  '휘펫',
]

def get_embedding(text: str):
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

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
    if any(k in t for k in ["대형","큼","큰","large"]): return "대형"
    if any(k in t for k in ["중간","중형","중","medium"]):       return "중형"
    if any(k in t for k in ["소형","작","small"]):          return "소형"
    return "알수없음"

def days_since(noticeSdt: Optional[str]) -> int:
    if not noticeSdt: return 0
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

def _text_has_keywords(text: Optional[str], keywords: list) -> bool:
    if not text:
        return False
    t = text.lower()
    for k in keywords:
        if k.lower() in t:
            return True
    return False

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _map_activity_level(level: str) -> float:
    return {"매우 활발함": 0.9, "보통": 0.6, "주로 실내 생활": 0.3}.get(level, 0.6)

def _map_care_time(cat: str) -> float:
    # numeric scale roughly matching previous mapping
    return {"10분 이하": 0.2, "30분": 0.4, "1시간": 0.6, "2시간 이상": 0.9}.get(cat, 0.5)

def _daily_home_time_adjustment(daily: str) -> float:
    # if user is home more, they can handle more active / higher care requirement pets
    if not daily: return 0.0
    return {"0~4시간": -0.15, "4~8시간": 0.0, "8~12시간": 0.10, "12시간 이상": 0.15}.get(daily, 0.0)
    
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
    return True

def size_match(pref: str, a_size: str) -> float:
    if pref in ("", "상관없음") or a_size == "알수없음": return 0.2
    if pref == a_size: return 1.0
    if (pref, a_size) in [("소형","중형"),("중형","소형"),("대형","중형"),("중형","대형")]:
        return 0.5
    return 0.0

def infer_activity_and_care(doc: Dict[str, Any]) -> Tuple[float, float]:
    # --- Defaults / baselines ---
    base_activity = 0.50  # neutral baseline
    grooming_base = 0.45  # neutral grooming need
    medical_need = 0.0

    # --- Breed / kind baseline heuristics (small set for quick biasing) ---
    kind = (doc.get("upKindNm") or doc.get("kindFullNm") or "").lower()
    if "리트리버" in kind or "보더콜리" in kind or "허스키" in kind:
        base_activity = 0.80
    elif "불독" in kind or "불도그" in kind or "퍼그" in kind:
        base_activity = 0.35
    elif "고양이" in kind or "한국 고양이" in kind or "고양" in kind:
        base_activity = 0.30

    # --- Size / weight based adjustments ---
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    rsize = a_size.lower()
    if "대형" in rsize or "큰" in rsize:
        base_activity += 0.05
    if "소형" in rsize or "작" in rsize:
        base_activity -= 0.05

    # --- Age based adjustments ---
    age_yrs = _age_in_years(doc.get("age") or "")
    if age_yrs is not None:
        if age_yrs <= 2.0:
            base_activity += 0.10   # younger => more active
        elif age_yrs >= 7.0:
            base_activity -= 0.20   # older => less active

    # --- Behavior / health textual signals (specialMark, health_impression, noticeable_features) ---
    text_fields = " ".join([
        (doc.get("specialMark") or ""),
        (doc.get("extractedFeature") or {}).get("health_impression", "") or "",
        (doc.get("extractedFeature") or {}).get("noticeable_features", "") or ""
    ]).lower()

    # positive activity indicators
    if _text_has_keywords(text_fields, ["활발", "사교", "사람 좋아", "장난", "에너지", "놀이"]):
        base_activity += 0.12

    # negative activity indicators
    if _text_has_keywords(text_fields, ["낯가림", "겁", "경계", "소심", "예민", "조심"]):
        base_activity -= 0.08

    # health-related indicators increase care need and reduce activity
    if _text_has_keywords(text_fields, ["병", "감염", "상처", "치료", "질환", "장애", "실명", "결손", "아픈", "염증"]):
        medical_need = max(medical_need, 0.7)
        base_activity -= 0.25

    # neuter effect: sterilized animals sometimes calmer
    neuter = (doc.get("neuterYn") or "").upper()
    if neuter == "Y":
        base_activity -= 0.05

    # --- Fur / grooming heuristics from extractedFeature ---
    fur_len = ((doc.get("extractedFeature") or {}).get("fur_length") or "").lower()
    fur_texture = ((doc.get("extractedFeature") or {}).get("fur_texture") or "").lower()

    if "긴" in fur_len or "장모" in fur_len:
        grooming_base = 0.9
    elif "짧" in fur_len:
        grooming_base = 0.2
    elif "중간" in fur_len:
        grooming_base = 0.5
    else:
        grooming_base = 0.45

    if _text_has_keywords(fur_texture, ["엉킴", "매트", "거친", "얽힘"]):
        grooming_base = min(1.0, grooming_base + 0.15)

    # skin / coat health issues bump grooming/medical need
    if _text_has_keywords(text_fields, ["피부", "털빠짐", "비듬", "기생충", "염증"]):
        grooming_base = min(1.0, grooming_base + 0.25)
        medical_need = max(medical_need, 0.6)

    # --- Compose care_score ---
    # care_score mixes grooming need and medical need; grooming weighted slightly more
    care_score = 0.6 * grooming_base + 0.4 * medical_need
    care_score = _clamp01(care_score)

    # --- Compose activity_score ---
    activity_score = _clamp01(base_activity)

    # Round to 3 decimals for neatness
    activity_score = round(activity_score, 3)
    care_score = round(care_score, 3)

    return (activity_score, care_score)

def compute_desired_activity_and_care(ans: Dict[str, Any]) -> Tuple[float, float]:
    if not ans:
        # defaults if no survey
        return 0.6, 0.5

    # base from explicit activityLevel field (primary signal)
    desired_activity = _map_activity_level(ans.get("activityLevel", ""))  # 0..1

    # dailyHomeTime increases possible activity tolerance
    daily = ans.get("dailyHomeTime", "")
    adj = _daily_home_time_adjustment(daily)
    if adj != 0.0:
        desired_activity = max(0.0, min(1.0, desired_activity + adj))

    # 가족(구성원) 수: 더 많은 가족은 활동적인 반려동물 허용 가능성 소폭 증가
    try:
        fc = int(ans.get("familyCount", 0))
        if fc >= 3:
            desired_activity = min(1.0, desired_activity + 0.05)
    except Exception:
        pass

    # 어린이나 노인이 있는 경우: 상대적으로 차분한 동물 선호 가능성 -> 활동성 조정
    if ans.get("hasChildOrElder") == "있음":
        desired_activity = max(0.0, desired_activity - 0.10)

    # 기존에 반려동물을 키워본 경험이 있다면 활동/케어 감내력 증가
    ph = ans.get("petHistory", "") or ""
    if ph and ("현재" in ph or "과거" in ph):
        desired_activity = min(1.0, desired_activity + 0.05)

    # 원하는 케어 시간: 기본 numeric target (from explicit careTime)
    desired_care = _map_care_time(ans.get("careTime", ""))
    # 예산/시간 여유가 많으면 더 많은 케어 허용 가능
    budget = ans.get("budget", "")
    if budget:
        # high budget category increases tolerance to higher care requirement
        if "700만원" in budget or "600만원" in budget or "500만원" in budget:
            desired_care = min(1.0, desired_care + 0.05)

    # preferredSize가 '상관없음'이 아니면 약간 영향
    if ans.get("preferredSize") and ans.get("preferredSize") != "":
        # 선호 크기가 클수록 케어 허용도가 약간 올라갈 수 있음 (단, 집 크기 고려는 별도)
        if ans.get("preferredSize") == "대형":
            desired_activity = min(1.0, desired_activity + 0.03)

    # clamp
    desired_activity = round(_clamp01(desired_activity), 3)
    desired_care = round(_clamp01(desired_care), 3)

    return desired_activity, desired_care


def compat_score(ans: Dict[str, Any], doc: Dict[str, Any]) -> float:
    if not ans: return 0.0
    score, wsum = 0.0, 0.0
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))

    # 크기 선호
    w = 0.35
    score += w * size_match(ans.get("preferredSize",""), a_size); wsum += w

    # 활동/케어 적합
    desired_active, care_time = compute_desired_activity_and_care(ans)
    a_act, a_care = infer_activity_and_care(doc)
    if doc.get("specialMark") and "활발" in doc.get("specialMark"):
        a_act = max(a_act, 0.8)
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
        "애교": ["애교", "교감", "스킨십", "사람 좋아", "친밀", "붙임성", "친화", "귀여운"],
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

def find_personality_matches(pref_pers: List[str], doc: Dict[str, Any]) -> List[str]:
    persona_text = " ".join([doc.get("specialMark","") or "", (doc.get("extractedFeature") or {}).get("noticeable_features","") or ""]).lower()
    TEMPERAMENT_MAP = {
        "차분": ["차분", "조용", "온순", "침착", "순함", "평온"],
        "활발": ["활발", "에너지", "활동", "발랄", "명랑"],
        "독립": ["독립", "자립", "혼자 잘"],
        "애교": ["애교", "교감", "사람 좋아", "붙임성", "친화", "귀여운"],
        "예민": ["예민", "민감", "경계"],
    }
    hits = []
    for pref in (pref_pers or []):
        if pref in TEMPERAMENT_MAP:
            if any(k in persona_text for k in TEMPERAMENT_MAP[pref]):
                hits.append(pref)
    return hits

def find_query_color_keywords(q: str) -> List[str]:
    if not q: return []
    ql = q.lower()
    kws = []
    # 단순 키워드 목록 (필요시 확장)
    KEYWORDS = ["검은색","검정","흰색","하얀","갈색","회색","치즈","베이지","크림",
                "점박이","얼룩"]
    for k in KEYWORDS:
        if k.lower() in ql:
            kws.append(k)
    return kws

def find_query_other_keywords(q: str) -> List[str]:
    if not q: return []
    ql = q.lower()
    kws = []
    # 단순 키워드 목록 (필요시 확장)
    KEYWORDS = ["대형","중형","소형","활발","활동","차분","애교","귀여운","아기",
                "짧은 털","긴 털","부드러운","거친","믹스","한국 고양이"]
    for k in KEYWORDS:
        if k.lower() in ql:
            kws.append(k)
    return kws

def build_reasons(survey: Dict[str, Any], q: str, meta: Dict[str, float], doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    reasons = []
    # 2) 임베딩 유사도
    sim = meta.get("sim", 0.0)
    if sim > 0.53:
        reasons.append({
            "type": "embedding",
            "label": f"쿼리/프로필 임베딩 유사도 높음 (sim={round(sim,3)})",
            "score": float(sim),
            "evidence": "",
            "reason": "설문과 유사함"
        })

    # 3) 크기 매칭
    pref_size = survey.get("preferredSize") if survey else ""
    a_size = parse_size((doc.get("extractedFeature") or {}).get("rough_size",""))
    s_size = size_match(pref_size or "", a_size)
    if s_size >= 0.8:
        reasons.append({
            "type": "size",
            "label": f"선호 크기와 일치: {a_size}",
            "score": float(s_size),
            "evidence": a_size,
            "reason": a_size
        })

    # 4) 성격 매칭(설문 -> doc.specialMark / noticeable_features)
    pref_pers = (survey.get("preferredPersonality") or []) if survey else []
    pers_hits = find_personality_matches(pref_pers, doc)
    if pers_hits:
        reasons.append({
            "type": "personality",
            "label": f"선호 성격과 부합: {', '.join(pers_hits)}",
            "score": 0.8,
            "evidence": (doc.get("specialMark") or "")[:120],
            "reason": ", ".join(pers_hits)
        })

    # 5) 활동/케어 적합
    if survey:
        desired_active, care_time = compute_desired_activity_and_care(survey)
        a_act, a_care = infer_activity_and_care(doc)
        act_match = 1 - min(1.0, abs(desired_active - a_act))
        care_match = 1 - min(1.0, abs(care_time - a_care))
        s_act = (act_match + care_match) / 2.0
        if desired_active > 0.7: reason = "활동적"
        elif desired_active < 0.4: reason = "차분함"
        else: reason = "활동성 보통"
        if s_act >= 0.8:
            reasons.append({
                "type": "activity_care",
                "label": f"활동성/케어가 잘 맞음",
                "score": float(s_act),
                "evidence": f"동물 활동성 추정:{a_act}, 사용자 기대:{desired_active}, 케어 시간 추정:{a_care}, 사용자 기대:{care_time}",
                "reason": reason
            })

    # 6) 위치 관련
    user_addr = survey.get("address") if survey else None
    loc_score = location_score(user_addr, doc.get("careAddr")) if user_addr else 0.0
    if loc_score and loc_score > 0.7:
        reasons.append({
            "type": "location",
            "label": f"지역 근접성: {round(loc_score,3)}",
            "score": float(loc_score),
            "evidence": doc.get("careAddr") or "",
            "reason": "가까운 지역"
        })

    # 7) 우선순위(장기 체류 / 건강 등)
    prio = meta.get("prio", 0.0)
    if prio and prio > 1.5:
        reasons.append({
            "type": "priority",
            "label": "우선 입양 권장(장기 보호/건강 고려)",
            "score": float(prio),
            "evidence": doc.get("specialMark") or "",
            "reason": "우선 입양 권장"
        })

    # 8) 키워드 매칭 근거 (쿼리에 등장하는 단어가 동물 문서의 필드에 등장하면 명시)
    q_c_keywords = find_query_color_keywords(q)
    matched_kw = []
    text_for_search = " ".join([
        doc.get("colorCd","") or "",
        doc.get("main_color","") or "",
    ]).lower()
    for kw in q_c_keywords:
        if kw.lower() in text_for_search:
            matched_kw.append(kw)

    q_o_keywords = find_query_other_keywords(q)
    text_for_search = " ".join([
        doc.get("kindNm","") or "",
        doc.get("specialMark","") or "",
        (doc.get("extractedFeature") or {}).get("noticeable_features","") or "",
        (doc.get("extractedFeature") or {}).get("fur_length","") or "",
        (doc.get("extractedFeature") or {}).get("fur_texture","") or "",
        (doc.get("extractedFeature") or {}).get("fur_pattern","") or "",
    ]).lower()
    for kw in q_o_keywords:
        if kw.lower() in text_for_search:
            matched_kw.append(kw)
    
    if matched_kw:
        reasons.append({
            "type": "keyword",
            "label": f"쿼리 키워드 일치: {', '.join(matched_kw)}",
            "score": 0.5,
            "evidence": ", ".join(matched_kw),
            "reason": ", ".join(matched_kw)
        })

    return reasons

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
    alpha = 0.2 if generic else 0.8  # generic일수록 설문 비중 증가
    w_sim, w_prio, w_loc = (0.8, 0.1, 0.1)

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
    beta = 0.35  # 0..1 — 0 = ignore field-level, 1 = only field-level

    # 2) 후보 생성
    species_list = extract_species(q)
    base_filter: Dict[str, Any] = {}
    if species_list:
        if len(species_list) == 1:
            base_filter["upKindNm"] = species_list[0]
        else:
            base_filter["upKindNm"] = {"$in": species_list}
    
    candidates: List[Tuple[float, Dict[str,Any], Dict[str,float]]] = []
    for doc in collection.find(base_filter):
        a_emb = np.array(doc.get("embedding") or [], dtype=np.float32)
        if a_emb.size == 0: continue

        sim_q = cosine_similarity(q_emb, a_emb) if q_emb is not None else 0.0
        sim_p = cosine_similarity(p_emb, a_emb) if p_emb is not None else 0.0
        sim_mix_baseline = (alpha * sim_q) + ((1 - alpha) * sim_p) if (q_emb is not None or p_emb is not None) else 0.0

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
        field_weights = {"color":0.32, "activity":0.38, "appearance":0.20, "personality":0.10}
        total_w = sum(field_weights.values())
        field_match_score = sum(field_scores[k] * (field_weights[k] / total_w) for k in field_scores.keys())

        sim_mix = (1.0 - beta) * sim_mix_baseline + beta * field_match_score
        if sim_mix <= 0: continue
        candidates.append((sim_mix, doc, {"sim_q": sim_q, "sim_p": sim_p, "field_match": field_scores}))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    pool = candidates[: max(100, body.limit * 10)]

    # 3) 설문 제약 필터
    filtered = []
    for sim, doc, meta_sim in pool:
        if survey and not survey_constraints(survey, doc):
            continue
        filtered.append((sim, doc, meta_sim))
    if not filtered:
        filtered = pool

    # 4) 호환/우선노출/위치기반 계산 + 최종 재랭킹
    scored: List[Tuple[float, Dict[str,Any], Dict[str,float]]] = []
    user_addr = survey.get("address") if survey else None
    for sim, doc, meta_sim in filtered:
        # comp = compat_score(survey, doc) if survey else 0.0 // 중복된 설문 정보 반영
        prio = priority_boost(doc)
        loc = location_score(user_addr, doc.get("careAddr"))
        final = w_sim*sim + w_prio*(prio/3.0) + w_loc*loc  # prio 0~3 → 0~1 정규화
        scored.append((final, doc, {"sim":sim, "prio":prio, "loc":loc, **meta_sim}))

    scored.sort(key=lambda x: x[0], reverse=True)
    topn = scored[: body.limit]

    results = []
    for (s, d, meta) in topn:
        reasons = build_reasons(survey, q, {"sim": meta.get("sim",0.0), "comp": meta.get("comp",0.0), "prio": meta.get("prio",0.0), "loc": meta.get("loc",0.0)}, d)
        results.append({
            "final": round(s,4),
            "sim": round(meta.get("sim",0.0),4),
            "comp": 0.0,
            "priority": round(meta.get("prio",0.0),3),
            "location": round(meta.get("loc",0.0),4),
            "field_match": meta.get("field_match", {}),
            "desertionNo": d.get("desertionNo"),
            "kindFullNm": d.get("kindFullNm"),
            "upKindNm": d.get("upKindNm"),
            "colorCd": d.get("colorCd"),
            "age": d.get("age"),
            "careAddr": d.get("careAddr"),
            "specialMark": d.get("specialMark"),
            "extractedFeature": d.get("extractedFeature"),
            "reasons": reasons
        })
    return results