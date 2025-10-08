# 설문조사 사용자 정보 API 명세

## 기본 정보
- **Prefix:** `/userinfo`
- **저장 경로:** `survey_data/{userId}.txt` (JSON 형식)

---

## 1. 설문 응답 저장

### `POST /userinfo/survey`

#### 요청 데이터(JSON, 모든 필드 필수)
```json
{
  "userId": "사용자명 또는 아이디",
  "address": "지역(예: 서울특별시 강남구 역삼동)",
  "residenceType": "거주 형태(아파트/단독주택/오피스텔/기타)",
  "hasPetSpace": "반려동물 공간 있음/없음",
  "familyCount": "가족/동거인 수(문자열)",
  "hasChildOrElder": "어린이나 노인 있음/없음",
  "dailyHomeTime": "평일 집에 머무는 시간",
  "hasAllergy": "알레르기 있음/없음",
  "allergyAnimal": "알레르기 동물(없으면 빈 문자열)",
  "activityLevel": "활동량(매우 활발함/보통/주로 실내)",
  "expectations": ["교감", "관리의 용이함", ...],       // 배열
  "favoriteAnimals": ["강아지", "고양이", ...],         // 배열
  "preferredSize": "소형/중형/대형/상관없음",
  "preferredPersonality": ["활발함", "애교 많음", ...], // 배열
  "careTime": "하루 케어 가능 시간",
  "budget": "월 지출 의향",
  "specialEnvironment": "기타 환경",
  "additionalNote": "기타 희망사항"
}
```

#### 응답
- 성공:
  ```json
  {
    "success": true,
    "msg": "설문 저장 완료"
  }
  ```
- 실패: (예외 발생 시 에러 메시지 반환)

---

## 2. 설문 응답 조회

### `GET /userinfo/survey/{userId}`

#### Path Parameter
- `userId`: 사용자명 또는 아이디

#### 응답
- 성공(설문 데이터 있음):
  ```json
  {
    "success": true,
    "data": {
      // SurveyRequest와 동일한 구조의 데이터
    }
  }
  ```
- 실패(설문 데이터 없음):
  ```json
  {
    "success": false,
    "msg": "설문 응답이 없습니다."
  }
  ```

---

## 기타

- 텍스트 파일 저장: `survey_data/{userId}.txt`에 JSON 전체 저장
- 모든 요청/응답은 JSON 형식
- 필드명, 타입은 위 SurveyRequest 모델과 일치해야 함

---