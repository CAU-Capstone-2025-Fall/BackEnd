import re
from typing import Optional, Tuple

# 시/도 표준화 매핑 (자주 쓰는 별칭 포함)
SIDO_ALIASES = {
    "서울": "서울특별시", "서울시": "서울특별시", "서울특별시": "서울특별시",
    "경기": "경기도", "경기도": "경기도",
    "인천": "인천광역시", "인천시": "인천광역시", "인천광역시": "인천광역시",
    "부산": "부산광역시", "부산광역시": "부산광역시",
    "대구": "대구광역시", "대구광역시": "대구광역시",
    "광주": "광주광역시", "광주광역시": "광주광역시",
    "대전": "대전광역시", "대전광역시": "대전광역시",
    "울산": "울산광역시", "울산광역시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시", "세종특별자치시": "세종특별자치시",
    "강원": "강원특별자치도", "강원도": "강원특별자치도", "강원특별자치도": "강원특별자치도",
    "충북": "충청북도", "충청북도": "충청북도",
    "충남": "충청남도", "충청남도": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도", "전북특별자치도": "전북특별자치도",
    "전남": "전라남도", "전라남도": "전라남도",
    "경북": "경상북도", "경상북도": "경상북도",
    "경남": "경상남도", "경상남도": "경상남도",
    "제주": "제주특별자치도", "제주도": "제주특별자치도", "제주특별자치도": "제주특별자치도",
}

CAPITAL_CLUSTER = {"서울특별시", "경기도", "인천광역시"}

# 간단한 이웃(인접) 시/도 클러스터. 정밀하지 않아도 "가까움" 힌트로 충분.
NEIGHBORS = {
    "서울특별시": {"경기도", "인천광역시"},
    "인천광역시": {"경기도", "서울특별시"},
    "경기도": {"서울특별시", "인천광역시", "강원특별자치도", "충청북도"},
    "강원특별자치도": {"경기도", "충청북도"},
    "충청북도": {"경기도", "강원특별자치도", "충청남도", "경상북도"},
    "충청남도": {"세종특별자치시", "대전광역시", "전라북도", "충청북도"},
    "세종특별자치시": {"충청남도", "대전광역시", "충청북도"},
    "대전광역시": {"충청남도", "충청북도"},
    "전북특별자치도": {"충청남도", "전라남도", "경상북도"},
    "전라남도": {"전북특별자치도", "광주광역시", "경상남도"},
    "광주광역시": {"전라남도"},
    "경상북도": {"충청북도", "전북특별자치도", "대구광역시", "경상남도"},
    "대구광역시": {"경상북도", "경상남도"},
    "경상남도": {"전라남도", "경상북도", "부산광역시", "울산광역시"},
    "부산광역시": {"경상남도", "울산광역시"},
    "울산광역시": {"경상남도", "부산광역시"},
    "제주특별자치도": set(),
}

def normalize_sido(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.strip()
    # 완전 일치
    if n in SIDO_ALIASES:
        return SIDO_ALIASES[n]
    # 접미사 제거 후 시도 판단
    # 예: "서울"만 들어온 경우
    for key in sorted(SIDO_ALIASES.keys(), key=len, reverse=True):
        if n.startswith(key):
            return SIDO_ALIASES[key]
    # "XX도", "XX시" 패턴
    m = re.match(r"^([가-힣]+?)(특별|광역|자치)?(시|도)$", n)
    if m:
        base = m.group(1)
        return SIDO_ALIASES.get(base, n)
    return n

def parse_address(addr: str) -> Tuple[Optional[str], Optional[str]]:
    """
    '서울특별시 강남구 역삼동...' → ('서울특별시', '강남구')
    '충청남도 부여군 임천면 ...' → ('충청남도', '부여군')
    """
    if not addr or not isinstance(addr, str):
        return None, None
    a = addr.strip()
    # 1) 정규식으로 시/도 + 시/군/구 추출
    m = re.search(r"(?P<sido>[가-힣]+(?:(?:특별|광역|자치)?(?:시|도)))\s+(?P<sigungu>[가-힣]+(?:시|군|구))", a)
    if m:
        sido = normalize_sido(m.group("sido"))
        sigungu = m.group("sigungu")
        return sido, sigungu
    # 2) 공백 기준 파싱 (앞 2토큰)
    parts = a.split()
    if len(parts) >= 2:
        sido = normalize_sido(parts[0])
        sigungu = parts[1] if re.search(r"(시|군|구)$", parts[1]) else None
        return sido, sigungu
    return None, None

def location_score(user_addr: Optional[str], care_addr: Optional[str]) -> float:
    """
    0.0 ~ 1.0:
      - 동일 시군구: 1.0
      - 동일 시도: 0.8
      - 수도권(서울/경기/인천) 상호: 0.6
      - 인접 시도: 0.5
      - 정보 부족/기타: 0.3
    """
    if not user_addr or not care_addr:
        return 0.3
    u_sido, u_sigungu = parse_address(user_addr)
    a_sido, a_sigungu = parse_address(care_addr)
    if not u_sido or not a_sido:
        return 0.3

    # 동일 시군구
    if u_sigungu and a_sigungu and (u_sido == a_sido) and (u_sigungu == a_sigungu):
        return 1.0
    # 동일 시도
    if u_sido == a_sido:
        return 0.8
    # 수도권 클러스터
    if u_sido in CAPITAL_CLUSTER and a_sido in CAPITAL_CLUSTER:
        return 0.6
    # 인접 시도
    if a_sido in NEIGHBORS.get(u_sido, set()):
        return 0.5
    return 0.3