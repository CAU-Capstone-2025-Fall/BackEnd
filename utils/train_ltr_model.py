import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pymongo import MongoClient
from tqdm import tqdm
import random

# 중요: recommend.py의 모든 로직을 가져와서 사용해야 합니다.
# 실제 프로젝트에서는 recommend 모듈을 import 하여 사용해야 합니다.
import routers.recommend as recommend
# from routers.recommend import (
#     build_profile_text,
#     is_generic_query,
#     extract_species,
#     survey_preferred_species,
#     user_to_4texts,
#     get_embeddings_batch,
#     _cached_text_embedding,
#     cosine_similarity,
#     compat_score_with_details,
#     priority_boost,
#     location_score,
#     compute_allergy_penalty,
#     compute_pet_conflict_penalty,
#     _clamp01,
#     _age_in_years, # train_ltr_model에서만 사용하는 일부 private 함수도 가져옵니다.
#     days_since,
# )

load_dotenv()
MONGODB_URI    = os.getenv("MONGODB_URI")
client_db = MongoClient(MONGODB_URI)
db = client_db["testdb"]
collection = db["abandoned_animals"]
survey_col = db["userinfo"]

# --- (변경) 학습에 사용할 다양하고 현실적인 사용자 쿼리 목록 ---
SAMPLE_QUERIES = [
    # 1. 종/크기 중심의 단순 쿼리 (5개)
    "강아지", "고양이 보여줘", "소형견", "대형견 입양", "중형 고양이",
    # 2. 외모/색상 중심의 구체적 쿼리 (5개)
    "흰색 말티즈", "치즈 태비 고양이", "검은색 강아지", "삼색 고양이", "베이지색 푸들",
    # 3. 성격/행동 중심의 쿼리 (7개)
    "사람 잘 따르는 강아지", "조용하고 얌전한 고양이", "활발하고 에너지 넘치는 친구",
    "애교 많은 개냥이", "혼자 있어도 얌전한 반려동물", "겁 없는 고양이", "산책 좋아하는 강아지",
    # 4. 나이/상태 중심의 쿼리 (3개)
    "아기 강아지 분양", "생후 1년 미만 어린 고양이", "나이가 좀 있는 차분한 개",
    # 5. 복합적/서술형 쿼리 (5개)
    "아파트에서 키우기 좋은 털 안빠지는 소형견", "얼굴 동글동글하고 귀여운 고양이",
    "첫 반려동물로 좋은 순한 강아지", "다리가 짧고 매력적인 강아지", "똑똑하고 훈련 잘되는 개",
    # 6. 추상적/일반적 쿼리 (5개)
    "나랑 잘 맞을 것 같은 동물 추천해줘", "귀여운 동물", "", "추천", "아무나 보여줘",
]


# LTR 모델에 사용할 피처 목록 (순서가 매우 중요!)
FEATURE_COLUMNS = [
    'sim_mix', 'compat_score', 'priority_score', 'location_score',
    'allergy_penalty', 'conflict_penalty', "sim_x_compat"
]

def extract_features_for_ltr(doc, meta):
    """ 학습 및 예측에 사용할 피처를 일관된 순서로 추출하는 함수 """
    sim_mix = meta.get('sim', 0.0)
    compat_score = meta.get('comp', 0.0)
    priority_score_raw = meta.get('prio', 0.0)
    priority_score = priority_score_raw / 3.0
    location_score = meta.get('loc', 0.0)
    allergy_penalty = meta.get('allergy_penalty', 0.0)
    conflict_penalty = meta.get('pet_conflict_penalty', 0.0)
    sim_x_compat = sim_mix * compat_score
    return [
        sim_mix, compat_score, priority_score, location_score,
        allergy_penalty, conflict_penalty, sim_x_compat,
    ]

def generate_training_data():
    """ 기존 휴리스틱 로직을 사용해 LTR 학습용 데이터를 생성하는 함수 """
    print("가상 사용자 요청을 시뮬레이션하여 학습 데이터를 생성합니다...")

    all_animals = list(collection.find({}))
    all_surveys = list(survey_col.find({}))
    if not all_surveys:
        raise ValueError("DB에 설문 데이터가 없습니다. populate_surveys.py를 먼저 실행해주세요.")
    print(f"전체 동물 수: {len(all_animals)}, 전체 설문 수: {len(all_surveys)}")

    training_data = []
    query_group_id = 0

    # (변경) 각 쿼리를 모든 설문과 조합하여 다양한 시나리오 생성
    pbar = tqdm(total=len(SAMPLE_QUERIES) * len(all_surveys), desc="Generating Data")
    for q in SAMPLE_QUERIES:
        for sdoc in all_surveys:
            survey = sdoc.get("answers") or sdoc
            
            # --- recommend_hybrid 함수의 전반부 로직 재사용 ---
            q_emb = recommend._cached_text_embedding(q) if q else None
            profile_text = recommend.build_profile_text(survey)
            p_emb = recommend._cached_text_embedding(profile_text) if profile_text else None
            
            generic = recommend.is_generic_query(q)
            alpha = 0.3 if generic else 0.7
            w_sim, w_comp, w_prio, w_loc = (0.4, 0.4, 0.15, 0.05) if generic else (0.6, 0.25, 0.1, 0.05)
            w_allergy, w_conflict = 0.6, 0.3
            
            # 각 동물에 대해 피처와 정답(휴리스틱 점수) 계산
            # 실제로는 모든 동물이 아닌 샘플링된 동물로 속도를 높일 수 있음
            animal_sample = random.sample(all_animals, k=min(len(all_animals), 200))
            for doc in animal_sample:
                a_emb = np.array(doc.get("embedding", []), dtype=np.float32)
                if a_emb.size == 0: continue

                sim_q = recommend.cosine_similarity(q_emb, a_emb) if q_emb is not None else 0.0
                sim_p = recommend.cosine_similarity(p_emb, a_emb) if p_emb is not None else 0.0
                sim_mix = (alpha * sim_q) + ((1 - alpha) * sim_p)
                
                comp, _ = recommend.compat_score_with_details(survey, doc)
                prio = recommend.priority_boost(doc)
                user_addr = survey.get("address")
                loc_raw = recommend.location_score(user_addr, doc.get("careAddr")) if user_addr else 0.0
                loc_component = (loc_raw - 0.5) * 2.0
                allergy_penalty = recommend.compute_allergy_penalty(survey, doc)
                conflict_penalty = recommend.compute_pet_conflict_penalty(survey, doc)

                final_heuristic_score = recommend._clamp01(
                    (w_sim * sim_mix) + (w_comp * comp) + (w_prio * (prio / 3.0)) + 
                    (w_loc * loc_component) - (w_allergy * allergy_penalty) - (w_conflict * conflict_penalty)
                )

                final_heuristic_score_int = int(final_heuristic_score * 10)
                
                meta = {
                    "sim": sim_mix, "comp": comp, "prio": prio, "loc": loc_raw,
                    "allergy_penalty": allergy_penalty, "pet_conflict_penalty": conflict_penalty
                }
                features = extract_features_for_ltr(doc, meta)
                
                training_data.append([query_group_id] + features + [final_heuristic_score_int])

            query_group_id += 1
            pbar.update(1)
    pbar.close()

    df = pd.DataFrame(training_data, columns=['query_id'] + FEATURE_COLUMNS + ['relevance'])
    return df

def train_model(df: pd.DataFrame):
    """ 생성된 데이터프레임으로 LGBMRanker 모델을 학습하고 저장 """
    print("LGBMRanker 모델 학습을 시작합니다...")
    group_counts = df['query_id'].value_counts().sort_index().tolist()
    X = df[FEATURE_COLUMNS]
    y = df['relevance']

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=200, learning_rate=0.05,
        num_leaves=31, random_state=42, n_jobs=-1,
    )
    ranker.fit(
        X, y, group=group_counts, eval_set=[(X, y)], eval_group=[group_counts],
        eval_at=[10], callbacks=[lgb.early_stopping(10, verbose=True)],
    )

    model_filename = "ltr_model.pkl"
    joblib.dump(ranker, model_filename)
    print(f"모델이 '{model_filename}' 파일로 저장되었습니다.")
    return ranker

if __name__ == "__main__":
    training_df = generate_training_data()
    print(f"\n총 학습 데이터 수: {len(training_df)}")
    print(f"총 쿼리 그룹 수: {training_df['query_id'].nunique()}")
    train_model(training_df)