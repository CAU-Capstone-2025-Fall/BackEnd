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
# 2) Summary 생성 함수 (interaction 포함)
# -------------------------------------------------------
async def generate_summary_text(prob, lime_top5, interaction_top3, raw):

    selected_human = []
    for key, val in raw.items():
        if val in [1, "1"]:
            selected_human.append(HUMAN_MAP.get(key, key))

    lime_human = [
        f"{HUMAN_MAP.get(feat, feat)}: {w:.4f}"
        for feat, w in lime_top5
    ]

    if interaction_top3:
        inter_human = [
            f"{' × '.join(groups)}: {score:.4f}" 
            for groups, score in interaction_top3
        ]
    else:
        inter_human = ["(상호작용 없음)"]

    prompt = f"""
다음 데이터를 기반으로 '반려동물 유기 위험도 분석 요약'을 작성해줘.

[유기 충동 확률]
{prob * 100:.1f}%

[LIME 영향 요인 Top 5]
{chr(10).join(lime_human)}

[상호작용 영향 요인 Top 3]
{chr(10).join(inter_human)}

[사용자 선택 정보]
- {chr(10).join(selected_human) if selected_human else "특이 선택 정보 없음"}

요구사항:
- 반드시 **하나의 제목과 아래 한칸 띄고 하나의 단락**으로 요약하여 작성할 것.
- 첫 제목: LIME + 상호작용 조합으로 사용자 친화적인 요약.
- 단락: 기반 개별 요인 과 상호작용(pairwise) 조합이 어떤 영향을 가지고 있는지 사용자 친화적인 설명.
- 문장은 자연스럽고 유창하게 작성.
- 제목과 단락 사이에 반드시 빈 줄 1개 삽입.
- 숫자는 직접 나열하지 말고 자연어로 의미만 해석.
- 확률 기준:
    - 30% 미만: 낮음
    - 30~60%: 보통
    - 60% 이상: 높음

출력: 제목과 한 단락으로 구성된 자연스러운 요약문
"""

    resp = await client.responses.create(
        model="gpt-5-nano",
        input=prompt,
        text={"verbosity": "low"},
        reasoning={"effort": "low"},
    )

    return resp.output_text

# -------------------------------------------------------
# 3) 행동 추천 생성 함수 (interaction 포함)
# -------------------------------------------------------
async def generate_recommendations_text(prob, lime_top5, interaction_top3, raw):
    prompt = f"""
다음 데이터를 기반으로 반려동물 유기 위험도를 낮추기 위한 
구체적이고 실천 가능한 행동 가이드 3개를 생성해줘.
단 , 각 행동 가이드는 매우 실행하기 쉽고 구체적인 가이드여야해.
예를 들어 반려동물과 더많은 시간을 보내기 말고,
반려동물과 하루에 한시간은 산책하기 같은 구체적인 행동 가이드여야해.

[유기 위험도]
{prob*100:.1f}%

[LIME 영향 요인 Top 5]
{lime_top5}

[상호작용 영향 Top 3]
{interaction_top3}

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
