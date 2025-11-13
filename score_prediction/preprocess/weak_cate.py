# ============================================================
# 🧩 Multidimensional Weak Label 생성 (gpt-4o-mini, '분류' 버전)
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

# ✅ [수정] 0-10점수(v3)와 구분을 위해 새 출력 파일명 지정
OUTPUT_PATH = os.path.join(DATA_DIR, "weak_labels_classification_v1.xlsx")
MODEL_NAME = "gpt-4o-mini"

# ------------------------------------------------------------
# 🧩 JSON 파싱 함수 (동일)
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
# 1️⃣ 시트 로드 (동일)
# ------------------------------------------------------------
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ {FILE_PATH} 파일을 찾을 수 없습니다.")

df_raw = pd.read_excel(FILE_PATH, sheet_name="마이크로데이터", header=[0, 1])
df_raw.columns = df_raw.columns.get_level_values(1)
df_code = pd.read_excel(FILE_PATH, sheet_name="코딩")

print(f"[INFO] 마이크로데이터 shape: {df_raw.shape}")
print(f"[INFO] 코딩 시트 shape: {df_code.shape}")

# ------------------------------------------------------------
# 2️⃣ 매핑 테이블 생성 (동일)
# ------------------------------------------------------------
df_code["ANSWER"] = df_code["ANSWER"].astype(str).str.strip()
df_code["VALUE"] = df_code["VALUE"].astype(str).str.strip()
df_code["QUESTION"] = df_code["QUESTION"].astype(str).str.strip()
if "SUB_QUESTION" in df_code.columns:
    df_code["SUB_QUESTION"] = df_code["SUB_QUESTION"].astype(str).str.strip()
else:
    df_code["SUB_QUESTION"] = ""

# (1) VALUE 매핑
code_map = {}
for _, row in df_code.dropna(subset=["COLUMNNAME", "ANSWER", "VALUE"]).iterrows():
    col = str(row["COLUMNNAME"]).strip()
    ans = str(row["ANSWER"]).strip()
    val = str(row["VALUE"]).strip()
    code_map[(col, ans)] = val
    try:
        f_ans = str(float(ans))
        code_map[(col, f_ans)] = val
        i_ans = str(int(float(ans)))
        code_map[(col, i_ans)] = val
    except:
        pass

# (2) QUESTION 매핑
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
# 3️⃣ 응답 복원 (동일)
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
# 4️⃣ ✅ [수정] 프롬프트 빌더 (0, 1, 2 숫자 분류 작업)
# ------------------------------------------------------------
def build_prompt(row):
    items = []
    for col, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        question_text = question_map.get(col, col)
        items.append(f"{question_text}: {val}")

    joined = "\n".join(items)

    # ✅ [수정] 0-10점수(회귀)가 아닌 [0, 1, 2](분류)를 요청
    prompt = f"""
You are a **strict and critical evaluator** analyzing survey responses about pet ownership.

Below is one respondent’s survey answers.

{joined}

Your task is to **classify this respondent conservatively** across six distinct psychological and behavioral dimensions
related to responsible pet ownership.

Be **objective, analytical, and slightly skeptical**. Avoid giving 'High' (2) ratings unless the evidence in the responses
strongly supports them.

Each dimension must receive one of three integer scores: **[0, 1, 2]**
- 0 = Low (poor or concerning tendencies)
- 1 = Medium (average or mixed tendencies)
- 2 = High (outstanding and clearly justified tendencies)

Dimensions:
1. empathy — emotional understanding and compassion toward animals
2. ethicality — moral responsibility and respect for animal welfare
3. self_control — impulse management and steady caregiving intention
4. care_environment — quality of financial, spatial, and time conditions for pet care
5. emotional_sensitivity — depth and expressiveness of emotional connection
6. behavioral_consistency — alignment between expressed values and actual behavioral intention

Return **only JSON** in this exact format, using **only** the integers 0, 1, or 2:
{{
  "empathy": 1,
  "ethicality": 2,
  "self_control": 0,
  "care_environment": 2,
  "emotional_sensitivity": 1,
  "behavioral_consistency": 1
}}
All values MUST be one of the integers 0, 1, or 2.
""".strip()
    return prompt
# ------------------------------------------------------------
# 5️⃣ ✅ [수정] GPT 호출 함수 (0, 1, 2 숫자 검증)
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

        # ---------------------------------
        # ✅ [수정] 문자열 검증 -> 0, 1, 2 정수 검증 로직
        # ---------------------------------
        score_keys = [
            "empathy", "ethicality", "self_control",
            "care_environment", "emotional_sensitivity", "behavioral_consistency"
        ]
        valid_scores = {0, 1, 2}

        for k in score_keys:
            if k in parsed:
                try:
                    # GPT가 "2", "1.0", 0 등 비표준화된 값을 줘도
                    # 정수 0, 1, 2로 변환
                    v = int(round(float(parsed[k])))
                    
                    if v not in valid_scores:
                        # GPT가 지시를 어기고 3이나 -1을 반환한 경우
                        print(f"⚠️ Invalid score '{v}' for {k}. Defaulting to 1 (Medium).")
                        parsed[k] = 1 # 안전한 기본값 'Medium'
                    else:
                        parsed[k] = v # 0, 1, 2
                except Exception:
                    # 그 외 모든 에러 (예: "High" 문자열, null)
                    parsed[k] = 1 # 안전한 기본값 'Medium'
            else:
                # 키가 누락된 경우
                parsed[k] = 1
        # ---------------------------------
        # [수정] 끝
        # ---------------------------------

        return parsed

    except Exception as e:
        print("⚠️ API Error:", e)
        return None
# ------------------------------------------------------------
# 6️⃣ 전체 응답 반복 처리 + 결과 저장 (동일)
# ------------------------------------------------------------
results = []
# ✅ df_text의 모든 행을 처리
sample_df = df_text

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Generating Weak Labels (Classification)"):
    parsed = get_multilabel(row.to_dict())
    if parsed:
        parsed["index"] = i
        results.append(parsed)
    else:
        print(f"⚠️ Row {i}: LLM 응답 없음")

# 결과 저장
df_result = pd.DataFrame(results)
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ [수정] OUTPUT_PATH가 이미 .xlsx를 가리키도록 수정
save_path = OUTPUT_PATH 
df_result.to_excel(save_path, index=False)

print(f"\n🎯 Weak classification labels saved → {save_path}")
print(f"🧾 총 생성된 결과 수: {len(df_result)}")