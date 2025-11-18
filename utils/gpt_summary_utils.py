# utils/gpt_summary_utils.py

import os

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# 1) 사람이 이해할 수 있는 feature dictionary
# -------------------------------------------------------
HUMAN_MAP = {
    "연령": "연령(나이)",
    "가족 구성원 수": "가족 구성원 수",
    "주택규모": "주택 규모(평수)",
    "월평균 가구소득": "월평균 가구 소득",
    "향후 반려동물 사육의향": "향후 사육 의향 정도",

    "성별_1": "남성",
    "성별_2": "여성",

    "주택형태_1": "아파트 거주",
    "주택형태_2": "단독/다가구 주택 거주",
    "주택형태_3": "연립/빌라/다세대 거주",
    "주택형태_4": "기타 주거형태",

    "화이트칼라": "화이트칼라 직군",
    "블루칼라": "블루칼라 직군",
    "자영업": "자영업",
    "비경제활동층": "비경제활동층",
    "기타": "기타 직업군",
}

# -------------------------------------------------------
# 2) Summary 생성 함수
# -------------------------------------------------------
async def generate_summary_text(prob, lime_top5, raw):
    """
    prob: float (0~1)
    lime_top5: list of (featName, weight)
    raw: dict (한글 key: value)
    """

    # 1) raw_input에서 value=1 인 것만 "선택된 속성" 리스트로
    selected_human = []
    for key, val in raw.items():
        if val in [1, "1"]:
            name = HUMAN_MAP.get(key, key)
            selected_human.append(name)

    # 2) LIME 영향 목록 → 사람이 읽을 수 있게 변환
    lime_human = []
    for feat, w in lime_top5:
        nm = HUMAN_MAP.get(feat, feat)
        lime_human.append(f"{nm}: {w:.4f}")

    # 3) 프롬프트 구성
    prompt = f"""
다음 정보를 바탕으로 '반려동물 유기 위험도 분석 요약'을 자연스럽고 정확하게 작성해줘.

[유기 충동 확률]
{prob * 100:.1f}%

[LIME 영향 요인 Top 5]
{chr(10).join(lime_human)}

[선택된 입력 정보]
- {chr(10).join(selected_human) if selected_human else "선택된 특성이 없음"}

조건:
- 확률이 30% 미만이면 "낮은 편", 30~60%면 "보통 수준", 60% 이상이면 "높은 편"이라고 판단.
- LIME에서 weight가 양수면 위험 증가 요인, 음수면 위험 감소 요인으로 해석.
- 성별/직업/주택형태 등은 raw_input 값이 1인 경우만 언급.
- 사실과 반대되는 문장은 절대 쓰지 말 것.
- 지나치게 길게 말하지 말고 1~2 문장으로 간결하게.

출력: 하나의 자연스러운 문단
    """

    resp = await client.responses.create(
        model="gpt-5-nano",
        input=prompt,
        text={"verbosity": "low"},
        reasoning={"effort": "low"},
    )

    return resp.output_text




async def generate_recommendations_text(prob, lime_top5, raw):
    prompt = f"""
다음 데이터를 기반으로 반려동물 유기 위험도를 낮추기 위한 
구체적이고 실천 가능한 행동 가이드 3개를 생성해줘.

[유기 위험도]
{prob*100:.1f}%

[영향 요인 Top 5]
{lime_top5}

[사용자 입력 정보]
{raw}

출력 규칙:
1. 절대 설명하지 말고 JSON만 출력해라.
2. JSON 구조는 반드시 아래 형식을 따라라:

{{
  "recommendations": [
    {{
      "title": "짧은 문장 제목",
      "detail": "구체적인 실행 가이드"
    }},
    {{
      "title": "...",
      "detail": "..."
    }},
    {{
      "title": "...",
      "detail": "..."
    }}
  ]
}}

3. title은 10자 이내, detail은 한 문장.
4. JSON 외 텍스트 절대 출력하지 말 것.
"""

    resp = await client.responses.create(
        model="gpt-5-nano",
        input=prompt,
        text={"verbosity": "low"}
    )
    return resp.output_text
