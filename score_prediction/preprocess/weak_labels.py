# ============================================================
# 🧩 Multidimensional Weak Label 생성 (gpt-4o-mini, with QUESTION & SUB_QUESTION mapping)
# ============================================================

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------
# ✅ 환경 설정
# ------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "../data"
FILE_PATH = os.path.join(DATA_DIR, "survey.xlsx")
OUTPUT_PATH = os.path.join(DATA_DIR, "weak_labels_multidim_v3.csv")
MODEL_NAME = "gpt-4o-mini"  # ✅ 안정적 JSON 출력 지원 모델

# ------------------------------------------------------------
# 🧩 JSON 파싱 함수
# ------------------------------------------------------------
def safe_parse_json(text: str):
    """GPT 출력에서 JSON 블록만 추출 후 파싱"""
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            clean_json = match.group(0)
            return json.loads(clean_json)
        else:
            return json.loads(text.strip())
    except Exception as e:
        print("⚠️ JSON Parse Error:", e)
        print("🧩 Raw content (first 300 chars):", text[:300])
        return None


# ------------------------------------------------------------
# 1️⃣ 시트 로드
# ------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ {FILE_PATH} 파일을 찾을 수 없습니다.")

df_raw = pd.read_excel(FILE_PATH, sheet_name="마이크로데이터", header=[0, 1])
df_raw.columns = df_raw.columns.get_level_values(1)
df_code = pd.read_excel(FILE_PATH, sheet_name="코딩")

print(f"[INFO] 마이크로데이터 shape: {df_raw.shape}")
print(f"[INFO] 코딩 시트 shape: {df_code.shape}")

# ------------------------------------------------------------
# 2️⃣ 매핑 테이블 생성 (QUESTION + SUB_QUESTION 우선)
# ------------------------------------------------------------
df_code["ANSWER"] = df_code["ANSWER"].astype(str).str.strip()
df_code["VALUE"] = df_code["VALUE"].astype(str).str.strip()
df_code["QUESTION"] = df_code["QUESTION"].astype(str).str.strip()
if "SUB_QUESTION" in df_code.columns:
    df_code["SUB_QUESTION"] = df_code["SUB_QUESTION"].astype(str).str.strip()
else:
    df_code["SUB_QUESTION"] = ""

# (1) VALUE 매핑 (COLUMNNAME + ANSWER → VALUE)
code_map = {}
for _, row in df_code.dropna(subset=["COLUMNNAME", "ANSWER", "VALUE"]).iterrows():
    col = str(row["COLUMNNAME"]).strip()
    ans = str(row["ANSWER"]).strip()
    val = str(row["VALUE"]).strip()
    code_map[(col, ans)] = val
    # 숫자형 허용 ("2" == "2.0")
    try:
        f_ans = str(float(ans))
        code_map[(col, f_ans)] = val
        i_ans = str(int(float(ans)))
        code_map[(col, i_ans)] = val
    except:
        pass

# (2) QUESTION 매핑 (SUB_QUESTION 우선, 없으면 QUESTION)
question_map = {}
for _, row in df_code.dropna(subset=["COLUMNNAME"]).iterrows():
    colname = str(row["COLUMNNAME"]).strip()
    sub_q = str(row.get("SUB_QUESTION", "")).strip()
    main_q = str(row.get("QUESTION", "")).strip()

    if sub_q and sub_q.lower() != "nan":
        question_map[colname] = sub_q
    elif main_q and main_q.lower() != "nan":
        question_map[colname] = main_q

print(f"[INFO] 코드 매핑 {len(code_map)}개, 질문 매핑 {len(question_map)}개 생성 완료\n")

# ------------------------------------------------------------
# 3️⃣ 응답 복원
# ------------------------------------------------------------
df_text = pd.DataFrame()

for col in df_raw.columns:
    responses = []
    for _, val in enumerate(df_raw[col]):
        if pd.isna(val):
            responses.append(None)
            continue

        val_str = str(val).strip()
        key = (col, val_str)
        mapped = code_map.get(key)

        # 숫자형 허용 (2 == 2.0)
        if mapped is None:
            try:
                val_float = str(float(val))
                mapped = code_map.get((col, val_float))
            except:
                mapped = None

        responses.append(mapped if mapped is not None else None)

    df_text[col] = responses

print(f"✅ 텍스트 변환 완료: {df_text.shape[0]}개 응답, {df_text.shape[1]}개 문항")

# ------------------------------------------------------------
# 4️⃣ 프롬프트 빌더 — COLUMNNAME을 QUESTION/SUB_QUESTION 텍스트로 매핑
# ------------------------------------------------------------
def build_prompt(row):
    items = []
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        question_text = question_map.get(col, col)
        items.append(f"{question_text}: {val}")

    joined = "\n".join(items)

    prompt = f"""
You are a **strict and critical evaluator** analyzing survey responses about pet ownership.

Below is one respondent’s survey answers.

{joined}

Your task is to **evaluate this respondent conservatively** across six distinct psychological and behavioral dimensions 
related to responsible pet ownership.

Be **objective, analytical, and slightly skeptical**. Avoid giving high scores unless the evidence in the responses 
strongly supports them.

Each dimension must receive an **integer score between 0 and 10**, where:
- 0–3 = poor or concerning tendencies
- 4–6 = average or mixed tendencies
- 7–8 = above average, but not perfect
- 9–10 = outstanding and clearly justified tendencies

Dimensions:
1. empathy — emotional understanding and compassion toward animals
2. ethicality — moral responsibility and respect for animal welfare
3. self_control — impulse management and steady caregiving intention
4. care_environment — quality of financial, spatial, and time conditions for pet care
5. emotional_sensitivity — depth and expressiveness of emotional connection
6. behavioral_consistency — alignment between expressed values and actual behavioral intention

Return **only JSON** in this exact format:
{{
반려동물 관련 정부의 중요 역할

}}
All scores MUST be integers between 0 and 10 inclusive.
""".strip()
    return prompt


# ------------------------------------------------------------
# 5️⃣ GPT 호출 함수 (안정형 + 0~10 점수 제한)
# ------------------------------------------------------------
def get_multilabel(row_dict):
    """GPT-4o-mini에 프롬프트를 보내고 JSON 응답 파싱"""
    prompt = build_prompt(row_dict)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )

        content = response.choices[0].message.content.strip()
        if not content or len(content) < 5:
            print("⚠️ Empty or invalid GPT response.")
            return None

        parsed = safe_parse_json(content)
        if parsed is None:
            print("⚠️ JSON 파싱 실패. GPT 응답 일부:")
            print(content[:300])
            return None

        # ✅ 점수 클램핑 (0~10 정수)
        score_keys = [
            "empathy", "ethicality", "self_control",
            "care_environment", "emotional_sensitivity", "behavioral_consistency"
        ]
        for k in score_keys:
            if k in parsed:
                try:
                    v = round(float(parsed[k]))
                    parsed[k] = int(max(0, min(10, v)))
                except:
                    parsed[k] = None

        return parsed

    except Exception as e:
        print("⚠️ API Error:", e)
        return None


# ------------------------------------------------------------
# 6️⃣ 전체 응답 반복 처리 + 결과 저장
# ------------------------------------------------------------
results = []
# ✅ df_text의 앞 3개 행만 샘플로 선택
sample_df = df_text

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Generating Weak Labels"):
    parsed = get_multilabel(row.to_dict())
    if parsed:
        parsed["index"] = i
        results.append(parsed)
    else:
        print(f"⚠️ Row {i}: LLM 응답 없음")

# 결과 저장
df_result = pd.DataFrame(results)
os.makedirs(DATA_DIR, exist_ok=True)
save_path = OUTPUT_PATH.replace(".csv", ".xlsx")
df_result.to_excel(save_path, index=False)

print(f"\n🎯 Weak multidimensional labels saved → {save_path}")
print(f"🧾 총 생성된 결과 수: {len(df_result)}")
