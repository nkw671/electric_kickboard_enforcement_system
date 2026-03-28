# 전동킥보드 단속 시스템 - 백엔드

## 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | Java 17 |
| 프레임워크 | Spring Boot 4.0 |
| 빌드 도구 | Gradle (Groovy) |
| 데이터베이스 | MySQL 8.0, Spring Data JPA |
| 라이브러리 | Lombok |
| API 문서화 | Swagger (Springdoc OpenAPI) |

-----

## 로컬 실행 방법

### 1\. 환경 설정 (MySQL)

로컬 환경에 MySQL 서버가 설치되어 있어야 하며, 아래 정보로 접속 가능해야 합니다.
(데이터베이스(`kickboard`)와 테이블은 서버 실행 시 `hibernate.ddl-auto=update` 옵션에 의해 자동 생성 및 갱신됩니다.)

`src/main/resources/application.properties` 확인:

```properties
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.url=jdbc:mysql://localhost:3306/kickboard?createDatabaseIfNotExist=true&serverTimezone=Asia/Seoul
spring.datasource.username=root
spring.datasource.password=본인의_DB_비밀번호
```

### 2\. 프로젝트 실행

IntelliJ IDEA에서 `BackApplication.java`의 `main` 메서드를 실행하거나, 터미널에서 아래 명령어를 입력합니다.

```bash
# 서버 실행 (http://localhost:8080)
./gradlew bootRun
```

-----

## 폴더 구조

```text
src/main/java/com/kickboard/back/
├── BackApplication.java          # 스프링 부트 앱 시작점
│
├── controller/                   # API 엔드포인트 및 HTTP 요청 처리
│   └── ViolationRecordController.java
│
├── service/                      # 비즈니스 로직 및 데이터 가공
│   └── ViolationRecordService.java
│
├── repository/                   # DB 접근 및 쿼리 실행 (Spring Data JPA)
│   └── ViolationRecordRepository.java
│
├── entity/                       #  DB 테이블과 1:1 매핑되는 엔티티
│   └── ViolationRecord.java
│
└── dto/                          # AI, 프론트와 주고받는 데이터 전송 객체
    ├── ViolationCreateRequest.java   # AI -> Back 수신용 DTO
    └── ViolationResponse.java        # Back -> Front 송신용 DTO
```

-----

## 데이터 흐름

```text
[ AI 영상 분석 서버 ]
        │ (POST JSON: 위반 유형, 사진 URL, 카메라 번호, 신뢰도)
        ▼
[ Controller ] ──(DTO)──▶ [ Service ] ──(Entity)──▶ [ Repository ]
                                                         │
                                                         ▼
[ 프론트엔드 웹 ] ◀──(DTO)── [ Service ] ◀──(Entity)── [ MySQL DB ]
        (GET JSON: 위반 목록, 통계)
```

-----

## API 명세서

서버 실행 후 아래 주소로 접속하면, 브라우저에서 직접 API를 테스트하고 전체 명세를 확인할 수 있습니다.

**[Swagger UI 접속: http://localhost:8080/swagger-ui/index.html](http://localhost:8080/swagger-ui/index.html)**

### 1\. 단속 데이터 수신 (AI -\> Back)

AI 서버에서 감지한 위반 데이터를 DB에 저장합니다.

  * **URL:** `POST /api/violations`
  * **Request Body (JSON):**
    ```json
    {
      "type": "헬멧 미착용",
      "image_url": "https://example.com/images/helmet_001.jpg",
      "camera": "CAM-01",
      "confidence": 94
    }
    ```
  * **Response:** `200 OK` ("단속 데이터 저장 성공")

### 2\. 단속 기록 조회 (Back -\> Front)

프론트엔드 메인 페이지 및 위반 기록 페이지에서 사용할 위반 목록을 최신순으로 반환합니다.

  * **URL:** `GET /api/violations?limit={number}`
  * **Query Parameter:**
      * `limit` (선택): 가져올 데이터 개수 (기본값: 10)
  * **Response:**
    ```json
    [
      {
        "id": 1,
        "type": "헬멧 미착용",
        "image_url": "https://example.com/images/helmet_001.jpg",
        "camera": "CAM-01",
        "confidence": 94,
        "timestamp": "2025-03-28 14:32:01"
      }
    ]
    ```

### 3\. 실시간 통계 조회 (Back -\> Front)

프론트엔드 하단 통계 카드에 표시될 오늘 발생한 위반 건수를 반환합니다.

  * **URL:** `GET /api/stats`
  * **Response:** *(현재 임시 데이터 반환 중, 실제 쿼리로 교체 예정)*
    ```json
    {
      "total": 12,
      "helmet": 7,
      "sidewalk": 3,
      "multiRider": 2
    }
    ```