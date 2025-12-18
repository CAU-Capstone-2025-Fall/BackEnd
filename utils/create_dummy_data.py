import os
import random
import uuid
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일에서 환경 변수 로드
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI가 .env 파일에 설정되지 않았습니다.")

# MongoDB 클라이언트 연결
client = MongoClient(MONGODB_URI)
db = client["testdb"]
collection = db["userinfo"]

# --- 설문 항목별 선택지 정의 ---
OPTIONS = {
    "address": ["서울특별시 강남구", "경기도 수원시 영통구", "부산광역시 해운대구", "대전광역시 유성구", "광주광역시 서구", "제주특별자치도 제주시", "강원특별자치도 춘천시", "충청북도 청주시", "전라남도 여수시"],
    "sex": ["남성", "여성"],
    "job": ["사무직", "전문직", "학생", "자영업", "주부", "프리랜서", "무직", "은퇴"],
    "residenceType": ["아파트", "빌라/연립주택", "단독주택", "오피스텔"],
    "hasPetSpace": ["있음", "없음"],
    "hasChildOrElder": ["있음", "없음"],
    "dailyHomeTime": ["0~4시간", "4~8시간", "8~12시간", "12시간 이상"],
    "hasAllergy": ["있음", "없음"],
    "activityLevel": ["매우 활발함", "보통", "주로 실내 생활"],
    "expectations": ["교감(애정 표현, 함께 놀기)", "활동적/에너지 넘침", "정서적 안정/위로", "조용한 동반자", "훈련 및 교육"],
    "favoriteAnimals": ["강아지", "고양이", "기타"],
    "preferredSize": ["소형", "중형", "대형", "상관없음"],
    "preferredPersonality": ["애교 많음", "활발함", "차분함", "독립적", "사교적", "겁 많음"],
    "careTime": ["10분 이하", "30분", "1시간", "2시간 이상"],
    "budget": ["100만원 미만", "100만원 ~ 199만원", "200만원 ~ 299만원", "300만원 ~ 399만원", "400만원 ~ 499만원", "500만원 이상"],
    "petHistory": ["반려동물을 키운 적 없다", "과거에 키운 경험이 있다", "현재 반려동물을 키우고 있다"],
    "houseSize": ["10평 미만", "10평 ~ 20평", "20평 ~ 30평", "30평 ~ 40평", "40평 이상"],
    "wantingPet": ["매우 의향이 있다", "의향이 있다", "보통이다"],
}

# --- 페르소나 기반 데이터 생성 함수 ---
def create_persona_data():
    surveys = []
    
    # 페르소나 1: 첫 반려동물을 꿈꾸는 1인 가구 (10개)
    for i in range(10):
        surveys.append({
            "userId": f"first_timer_{i+1}",
            "age": str(random.randint(22, 35)), "sex": random.choice(OPTIONS["sex"]),
            "job": random.choice(["사무직", "학생", "프리랜서"]),
            "residenceType": random.choice(["오피스텔", "빌라/연립주택"]), "houseSize": random.choice(["10평 미만", "10평 ~ 20평"]),
            "hasPetSpace": "없음", "familyCount": "1", "hasChildOrElder": "없음",
            "dailyHomeTime": random.choice(["4~8시간", "8~12시간"]),
            "petHistory": "반려동물을 키운 적 없다", "currentPets": [],
            "hasAllergy": "없음", "allergyAnimal": "",
            "activityLevel": random.choice(["보통", "주로 실내 생활"]), "careTime": random.choice(["30분", "1시간"]),
            "preferredSize": random.choice(["소형", "중형"]), "favoriteAnimals": [random.choice(["강아지", "고양이"])],
            "preferredPersonality": random.sample(["애교 많음", "차분함", "독립적"], k=random.randint(1, 2)),
            "expectations": random.sample(["교감(애정 표현, 함께 놀기)", "정서적 안정/위로"], k=random.randint(1, 2)),
            "budget": random.choice(["100만원 ~ 199만원", "200만원 ~ 299만원"]),
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "", "additionalNote": "", "wantingPet": "매우 의향이 있다"
        })

    # 페르소나 2: 아이와 함께할 동물을 찾는 가족 (10개)
    for i in range(10):
        surveys.append({
            "userId": f"family_user_{i+1}",
            "age": str(random.randint(35, 48)), "sex": random.choice(OPTIONS["sex"]),
            "job": random.choice(["사무직", "전문직", "자영업"]),
            "residenceType": random.choice(["아파트", "단독주택"]), "houseSize": random.choice(["30평 ~ 40평", "40평 이상"]),
            "hasPetSpace": random.choice(["있음", "없음"]), "familyCount": str(random.randint(3, 5)), "hasChildOrElder": "있음",
            "dailyHomeTime": random.choice(["4~8시간", "8~12시간"]),
            "petHistory": random.choice(["과거에 키운 경험이 있다", "반려동물을 키운 적 없다"]), "currentPets": [],
            "hasAllergy": "없음", "allergyAnimal": "",
            "activityLevel": random.choice(["매우 활발함", "보통"]), "careTime": random.choice(["1시간", "2시간 이상"]),
            "preferredSize": random.choice(["중형", "대형"]), "favoriteAnimals": ["강아지"],
            "preferredPersonality": random.sample(["활발함", "사교적", "애교 많음"], k=random.randint(1, 2)),
            "expectations": ["활동적/에너지 넘침", "교감(애정 표현, 함께 놀기)"],
            "budget": random.choice(["300만원 ~ 399만원", "500만원 이상"]),
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "", "additionalNote": "아이들과 잘 지내는 동물이면 좋겠습니다.", "wantingPet": "매우 의향이 있다"
        })

    # 페르소나 3: 숙련된 고양이 집사 (10개)
    for i in range(10):
        has_current_cat = random.choice([True, False])
        surveys.append({
            "userId": f"cat_lover_{i+1}",
            "age": str(random.randint(28, 45)), "sex": random.choice(OPTIONS["sex"]),
            "job": random.choice(["프리랜서", "전문직", "사무직"]),
            "residenceType": random.choice(["아파트", "빌라/연립주택"]), "houseSize": random.choice(["10평 ~ 20평", "20평 ~ 30평"]),
            "hasPetSpace": "없음", "familyCount": str(random.randint(1, 2)), "hasChildOrElder": "없음",
            "dailyHomeTime": random.choice(["8~12시간", "12시간 이상"]),
            "petHistory": "현재 반려동물을 키우고 있다" if has_current_cat else "과거에 키운 경험이 있다", 
            "currentPets": ["고양이"] if has_current_cat else [],
            "hasAllergy": random.choice(["있음", "없음"]), "allergyAnimal": "강아지" if surveys[-1]["hasAllergy"] == "있음" else "",
            "activityLevel": "주로 실내 생활", "careTime": random.choice(["30분", "1시간"]),
            "preferredSize": random.choice(["소형", "중형", "상관없음"]), "favoriteAnimals": ["고양이"],
            "preferredPersonality": random.sample(["독립적", "애교 많음", "차분함"], k=random.randint(1, 2)),
            "expectations": ["교감(애정 표현, 함께 놀기)", "정서적 안정/위로"],
            "budget": random.choice(["200만원 ~ 299만원", "400만원 ~ 499만원"]),
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "", "additionalNote": "고양이의 습성을 잘 이해하고 있습니다.", "wantingPet": "매우 의향이 있다"
        })

    # 페르소나 4: 차분한 노후를 함께할 동반자를 찾는 은퇴자 (10개)
    for i in range(10):
        surveys.append({
            "userId": f"senior_companion_{i+1}",
            "age": str(random.randint(60, 75)), "sex": random.choice(OPTIONS["sex"]),
            "job": "은퇴",
            "residenceType": random.choice(["아파트", "단독주택"]), "houseSize": random.choice(["20평 ~ 30평", "30평 ~ 40평"]),
            "hasPetSpace": "없음", "familyCount": str(random.randint(1, 2)), "hasChildOrElder": "없음",
            "dailyHomeTime": "12시간 이상",
            "petHistory": "과거에 키운 경험이 있다", "currentPets": [],
            "hasAllergy": "없음", "allergyAnimal": "",
            "activityLevel": "주로 실내 생활", "careTime": "1시간",
            "preferredSize": "소형", "favoriteAnimals": [random.choice(["강아지", "고양이"])],
            "preferredPersonality": ["차분함", "애교 많음"],
            "expectations": ["정서적 안정/위로", "조용한 동반자"],
            "budget": "300만원 ~ 399만원",
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "주로 집에 머무릅니다.", "additionalNote": "", "wantingPet": "의향이 있다"
        })
        
    # 페르소나 5: 마당 있는 집의 대형견 애호가 (5개)
    for i in range(5):
        surveys.append({
            "userId": f"large_dog_lover_{i+1}",
            "age": str(random.randint(38, 55)), "sex": random.choice(OPTIONS["sex"]),
            "job": random.choice(["자영업", "전문직"]),
            "residenceType": "단독주택", "houseSize": "40평 이상",
            "hasPetSpace": "있음", "familyCount": str(random.randint(2, 4)), "hasChildOrElder": random.choice(["있음", "없음"]),
            "dailyHomeTime": random.choice(["4~8시간", "8~12시간"]),
            "petHistory": "과거에 키운 경험이 있다", "currentPets": [],
            "hasAllergy": "없음", "allergyAnimal": "",
            "activityLevel": "매우 활발함", "careTime": "2시간 이상",
            "preferredSize": "대형", "favoriteAnimals": ["강아지"],
            "preferredPersonality": ["활발함", "사교적"],
            "expectations": ["활동적/에너지 넘침", "훈련 및 교육"],
            "budget": "500만원 이상",
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "넓은 마당이 있음.", "additionalNote": "대형견 훈련 경험 많음.", "wantingPet": "매우 의향이 있다"
        })

    # 페르소나 6: 특정 제약 조건을 가진 입양 희망자 (5개)
    for i in range(5):
        has_allergy = random.choice([True, False])
        surveys.append({
            "userId": f"special_case_{i+1}",
            "age": str(random.randint(30, 40)), "sex": random.choice(OPTIONS["sex"]),
            "job": "사무직",
            "residenceType": "아파트", "houseSize": "20평 ~ 30평",
            "hasPetSpace": "없음", "familyCount": "2", "hasChildOrElder": "없음",
            "dailyHomeTime": "4~8시간",
            "petHistory": "반려동물을 키운 적 없다", "currentPets": [],
            "hasAllergy": "있음" if has_allergy else "없음", 
            "allergyAnimal": "고양이" if has_allergy else "",
            "activityLevel": "보통", "careTime": "1시간",
            "preferredSize": "소형", "favoriteAnimals": ["강아지"] if has_allergy else ["고양이"],
            "preferredPersonality": ["차분함", "애교 많음"],
            "expectations": ["정서적 안정/위로", "교감(애정 표현, 함께 놀기)"],
            "budget": "300만원 ~ 399만원",
            "address": random.choice(OPTIONS["address"]), "specialEnvironment": "알러지가 있어 청결에 매우 신경 씀." if has_allergy else "", 
            "additionalNote": "", "wantingPet": "의향이 있다"
        })
    
    return surveys


def main():
    """
    메인 실행 함수
    """
    
    print("다양한 페르소나 기반 설문 데이터 50개 생성을 시작합니다...")
    survey_data = create_persona_data()
    
    print(f"총 {len(survey_data)}개의 설문 데이터가 생성되었습니다.")
    
    if not survey_data:
        print("생성된 데이터가 없습니다. 스크립트를 종료합니다.")
        return
        
    print("MongoDB에 데이터를 삽입합니다...")
    try:
        collection.insert_many(survey_data, ordered=False)
        print("데이터 삽입이 성공적으로 완료되었습니다!")
        print(f"현재 'userinfo' 컬렉션의 문서 수: {collection.count_documents({})}")
    except Exception as e:
        print(f"데이터 삽입 중 오류가 발생했습니다: {e}")
    finally:
        client.close()
        print("MongoDB 연결이 종료되었습니다.")


if __name__ == "__main__":
    main()