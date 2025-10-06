# abandoned_animal db API 명세

---

## Base URL
```
http://<server-host>:<port>
```

---

## 1. Animal 모델 (요청/응답에서 사용)
```json
{
  "desertionNo": "string",
  "happenDt": "string",
  "happenPlace": "string",
  "kindFullNm": "string",
  "upKindCd": "string",
  "upKindNm": "string",
  "kindCd": "string",
  "kindNm": "string",
  "colorCd": "string",
  "age": "string",
  "weight": "string",
  "noticeNo": "string",
  "noticeSdt": "string",
  "noticeEdt": "string",
  "popfile1": "string",
  "popfile2": "string",
  "processState": "string",
  "sexCd": "string",
  "neuterYn": "string",
  "specialMark": "string",
  "sfeSoci": "string",
  "sfeHealth": "string",
  "etcBigo": "string",
  "careRegNo": "string",
  "careNm": "string",
  "careTel": "string",
  "careAddr": "string",
  "careOwnerNm": "string",
  "orgNm": "string",
  "updTm": "string",
  "vaccinationChk": "string",
  "healthChk": "string"
}
```
---

## 2. 동물 등록 (Create Animal)
- **URL**: `/animal`
- **Method**: `POST`
- **Body (JSON)**:
  - [Animal 모델] 사용
  - 필수: `desertionNo`
- **입력 예시**:
  ```json
  {
    "desertionNo": "12345678",
    "happenDt": "20250917",
    ...
  }
  ```
- **성공 응답**:  
  ```json
  {
    "id": "<생성된 MongoDB document의 id>"
  }
  ```
- **에러 응답**:
  - 400: 이미 desertionNo가 존재할 때
  ```json
  {
    "detail": "Animal already exists."
  }
  ```

---

## 3. 동물 전체 조회 및 조건 검색 (Read Animals)
- **URL**: `/animal`
- **Method**: `GET`
- **Query Params (선택적)**:
  - `start_date` (string, YYYY-MM-DD): 조회 시작 날짜
  - `end_date` (string, YYYY-MM-DD): 조회 끝 날짜
  - `happen_place` (string): 발생 장소
  - `upkind_nm` (string): 대분류 이름
  - `kind_nm` (string): 세부 품종 이름
  - `sex_cd` (string): 성별 코드
  - `care_name` (string): 보호소 이름
  - `org_name` (string): 시군구 정보
  - `limit` (int, default=10): 한 번에 반환할 최대 개수
  - `skip` (int, default=0): 건너뛸 개수 (페이지네이션)
- **입력 예시**:
  ```
  http://<server-host>:<port>/animal?start_date=20250901&upkind_nm=개
  ```
- **성공 응답**:  
  ```json
  [
    {
      "desertionNo": "...",
      "happenDt": "20250901",
      "upKindNm": "개",
      ...
      "id": "MongoDB ObjectId"
    },
    ...
  ]
  ```

---

## 4. 특정 동물 조회 (Read Animal by desertionNo)
- **URL**: `/animal/{desertion_no}`
- **Method**: `GET`
- **Path Param**:
  - `desertion_no` (string): 유기번호
- **입력 예시**:
  ```
  http://<server-host>:<port>/animal/12345678
  ```
- **성공 응답**:
  ```json
  {
    "desertionNo": "12345678",
    "happenDt": "...",
    ...
    "id": "MongoDB ObjectId"
  }
  ```
- **에러 응답**:
  - 404: 해당 동물이 없을 때
  ```json
  {
    "detail": "Animal not found"
  }
  ```

---

## 5. 동물 정보 수정 (Update Animal)
- **URL**: `/animal/{desertion_no}`
- **Method**: `PUT`
- **Path Param**:
  - `desertion_no` (string): 유기번호
- **Body (JSON)**:
  - [Animal 모델] 사용
- **입력 예시**:
  ```
  http://<server-host>:<port>/animal/12345678
  ```
  ```json
  {
    "desertionNo": "12345678",
    "happenDt": "...",
    ...
  }
  ```
  - **성공 응답**:
  ```json
  {
    "desertionNo": "...",
    ...
    "id": "MongoDB ObjectId"
  }
  ```
- **에러 응답**:
  - 404: 해당 동물이 없을 때
  ```json
  {
    "detail": "Animal not found"
  }
  ```

---

## 6. 동물 삭제 (Delete Animal)
- **URL**: `/animal/{desertion_no}`
- **Method**: `DELETE`
- **Path Param**:
  - `desertion_no` (string): 유기번호
- **성공 응답**:
  ```json
  {
    "msg": "Animal {desertion_no} deleted."
  }
  ```
- **에러 응답**:
  - 404: 해당 동물이 없을 때
  ```json
  {
    "detail": "Animal not found"
  }
  ```

---

## 7. 공통사항

- 모든 응답에 `id` 필드는 MongoDB의 ObjectId를 문자열로 반환
- 페이지네이션: `limit`과 `skip`을 조합하여 원하는 만큼 데이터 조회 가능

---

## 8. 사용 예시

### 전체 조회 (50개씩 페이지네이션)
```
GET /animal?limit=50&skip=0
GET /animal?limit=50&skip=50
```

### 특정 보호소, 성별 조회
```
GET /animal?care_name=서울동물보호소&sex_cd=M
```

### 동물 등록
```
POST /animal
Body: { ...Animal 모델 내용... }
```

### 동물 삭제
```
DELETE /animal/1234567
```