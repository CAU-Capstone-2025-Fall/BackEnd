import json
from openai import OpenAI
from pathlib import Path

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI()  # 환경변수 OPENAI_API_KEY 사용

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

KEYWORDS = sorted(set(k.strip() for k in KEYWORDS if k.strip()))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = PROJECT_ROOT / "keyword_embeddings.json"

def main():
    try:
        with open(EMB_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[EMB] loaded {len(existing)} existing keyword embeddings from {EMB_PATH}")
    except FileNotFoundError:
        existing = {}
        print(f"[EMB] no existing {EMB_PATH}, starting fresh")

    out = dict(existing)
    new_count = 0

    for kw in KEYWORDS:
        if kw in existing:
            print(f"[EMB] skip '{kw}' (already exists)")
            continue

        print(f"[EMB] embedding '{kw}' ...")
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=kw
        ).data[0].embedding
        out[kw] = emb
        new_count += 1

    # 통합해서 다시 저장
    with open(EMB_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[EMB] saved {len(out)} keyword embeddings to {EMB_PATH} (new: {new_count})")


if __name__ == "__main__":
    main()
