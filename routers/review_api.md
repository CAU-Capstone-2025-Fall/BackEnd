---

# Review CRUD API 명세서

> 파일 기반(`reviews.json`) 리뷰 관리 API
> 이미지 업로드는 서버의 `uploads/` 폴더에 저장되며, `/static/` 경로로 접근 가능함.
> 인증은 `/auth/protected` 라우트를 이용한 **쿠키 기반 로그인 세션**으로 이루어짐.

---

## Base URL

```
http://<EC2-PUBLIC-IP>:<PORT>
```

예시:

```
http://3.36.xx.xx:8000
```

---

## Review 데이터 구조

```json
{
  "_id": "string (uuid)", // 리뷰 고유 ID
  "title": "string", // 제목
  "body": "string", // 본문 내용
  "images": ["string"], // 이미지 URL 배열 (/static/...)
  "authorId": "string", // 작성자 아이디
  "authorName": "string", // 작성자 표시 이름
  "createdAt": "string (ISO 8601)", // 작성일시
  "updatedAt": "string (ISO 8601)" // 수정일시
}
```

> - 모든 리뷰는 `reviews.json`에 배열 형태로 저장됩니다.
> - 각 리뷰는 `_id` 필드로 구분됩니다. (`uuid4()` 자동 생성)

---

## 후기 목록 조회 (List Reviews)

- **URL**: `/reviews`
- **Method**: `GET`
- **Query Params**

  | 이름    | 타입 | 기본값 | 설명           |
  | ------- | ---- | ------ | -------------- |
  | `skip`  | int  | 0      | 건너뛸 개수    |
  | `limit` | int  | 10     | 최대 조회 개수 |

### 성공 응답

```json
{
  "items": [
    {
      "_id": "fcdfaa9d-d2c1-4a2f-a7c3-df92b51935a7",
      "title": "첫 후기",
      "body": "이 강아지는 정말 귀여워요!",
      "images": ["/static/1696612345_abc123.jpg"],
      "authorId": "user1",
      "authorName": "user1",
      "createdAt": "2025-10-06T12:40:00Z",
      "updatedAt": "2025-10-06T12:40:00Z"
    }
  ],
  "total": 1
}
```

---

## 후기 단건 조회 (Get Review)

- **URL**: `/reviews/{rid}`
- **Method**: `GET`
- **Path Param**

  | 이름  | 타입   | 설명         |
  | ----- | ------ | ------------ |
  | `rid` | string | 리뷰의 `_id` |

### 성공 응답

```json
{
  "_id": "fcdfaa9d-d2c1-4a2f-a7c3-df92b51935a7",
  "title": "첫 후기",
  "body": "이 강아지는 정말 귀여워요!",
  "images": ["/static/1696612345_abc123.jpg"],
  "authorId": "user1",
  "authorName": "user1",
  "createdAt": "2025-10-06T12:40:00Z",
  "updatedAt": "2025-10-06T12:40:00Z"
}
```

### 에러 응답

```json
{ "detail": "존재하지 않는 글" }
```

---

## 후기 작성 (Create Review)

- **URL**: `/reviews`
- **Method**: `POST`
- **인증 필요**: (로그인 세션 쿠키 `session`)
- **Body**: `multipart/form-data`

  | 필드명  | 타입   | 필수 | 설명                     |
  | ------- | ------ | ---- | ------------------------ |
  | `title` | string | O    | 제목                     |
  | `body`  | string | X    | 본문                     |
  | `files` | file[] | X    | 이미지 여러 장 첨부 가능 |

### 성공 응답

```json
{
  "_id": "4b24de26-3e2e-4f57-a0b0-6fa7f91f0873",
  "title": "새로운 후기",
  "body": "강아지 너무 귀엽네요",
  "images": ["/static/1696612501_4b24de26.jpg"],
  "authorId": "user1",
  "authorName": "user1",
  "createdAt": "2025-10-06T13:00:00Z",
  "updatedAt": "2025-10-06T13:00:00Z"
}
```

### 에러 응답

```json
{ "detail": "로그인 필요" }
```

---

## 후기 수정 (Update Review)

- **URL**: `/reviews/{rid}`
- **Method**: `PATCH`
- **인증 필요**: (로그인 세션 쿠키 `session`)
- **Body**: `multipart/form-data`

  | 필드명  | 타입   | 필수 | 설명                                      |
  | ------- | ------ | ---- | ----------------------------------------- |
  | `title` | string | X    | 수정할 제목                               |
  | `body`  | string | X    | 수정할 본문                               |
  | `files` | file[] | X    | 새 이미지 첨부 시 기존 이미지 전부 교체됨 |

### 성공 응답

```json
{
  "_id": "4b24de26-3e2e-4f57-a0b0-6fa7f91f0873",
  "title": "수정된 제목",
  "body": "본문도 수정됨",
  "images": ["/static/1696613000_newimg.jpg"],
  "authorId": "user1",
  "authorName": "user1",
  "createdAt": "2025-10-06T13:00:00Z",
  "updatedAt": "2025-10-06T13:10:00Z"
}
```

### 에러 응답

- 작성자가 아닌 경우:

  ```json
  { "detail": "작성자만 수정 가능" }
  ```

- 존재하지 않는 글:

  ```json
  { "detail": "존재하지 않는 글" }
  ```

---

## 후기 삭제 (Delete Review)

- **URL**: `/reviews/{rid}`
- **Method**: `DELETE`
- **인증 필요**: (로그인 세션 쿠키 `session`)
- **Path Param**

  | 이름  | 타입   | 설명                |
  | ----- | ------ | ------------------- |
  | `rid` | string | 삭제할 리뷰의 `_id` |

### 성공 응답

```json
{ "ok": true }
```

### 에러 응답

- 존재하지 않는 글:

  ```json
  { "detail": "존재하지 않는 글" }
  ```

- 작성자가 아닐 때:

  ```json
  { "detail": "작성자만 삭제 가능" }
  ```

---

## 파일 저장 구조 (EC2 서버)

| 항목          | 경로                            | 설명                                    |
| ------------- | ------------------------------- | --------------------------------------- |
| 후기 저장     | `/home/ubuntu/app/reviews.json` | 모든 리뷰 데이터                        |
| 이미지 저장   | `/home/ubuntu/app/uploads/`     | 실제 업로드된 파일                      |
| 정적 서빙 URL | `/static/<파일명>`              | FastAPI가 `/uploads` 폴더를 외부 공개함 |

---

## 예시 요청 (Postman or Fetch)

```bash
POST /reviews
Content-Type: multipart/form-data
Cookie: session=<세션토큰>

Body:
  title="후기 제목"
  body="본문입니다"
  files=@cat.jpg
```

응답:

```json
{
  "_id": "af12b2c7-8a4f-4b32-933e-9a4c8e8fdd20",
  "title": "후기 제목",
  "body": "본문입니다",
  "images": ["/static/1696614000_af12b2c7.jpg"],
  "authorId": "tester",
  "createdAt": "2025-10-06T13:20:00Z",
  "updatedAt": "2025-10-06T13:20:00Z"
}
```

---

## 요약

| 기능           | URL              | Method   | 설명               |
| -------------- | ---------------- | -------- | ------------------ |
| 후기 목록 조회 | `/reviews`       | `GET`    | 모든 후기 목록     |
| 후기 단건 조회 | `/reviews/{rid}` | `GET`    | 특정 후기 상세     |
| 후기 작성      | `/reviews`       | `POST`   | 새 후기 작성       |
| 후기 수정      | `/reviews/{rid}` | `PATCH`  | 자신의 후기만 수정 |
| 후기 삭제      | `/reviews/{rid}` | `DELETE` | 자신의 후기만 삭제 |

---
