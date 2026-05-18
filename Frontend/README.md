# Frontend - 전동킥보드 단속 시스템

## 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | JavaScript (ES2020+) |
| 프레임워크 | React 18 |
| 빌드 도구 | Vite |
| 스타일 | CSS Modules |
| 라우팅 | React Router v6 |
| 통신 | Fetch API (폴링), SSE (실시간 알림) |

---

## 로컬 실행 방법

### 1. 의존성 설치
```bash
cd Frontend
npm install
```

### 2. 백엔드 서버 주소 확인
`vite.config.js`에서 백엔드 주소를 확인합니다.

```js
// 같은 컴퓨터에서 백엔드 실행 시 (기본값)
proxy: {
  '/api': 'http://localhost:8080',
}

// 다른 컴퓨터에서 실행 중인 경우 해당 IP로 변경
proxy: {
  '/api': 'http://192.168.0.xx:8080',
}
```

### 3. 개발 서버 실행
```bash
npm run dev
```
브라우저에서 `http://localhost:5173` 접속

---

## 폴더 구조

```
src/
├── components/
│   ├── Layout.jsx            # 공통 헤더 및 네비게이션
│   ├── StatCard.jsx          # 통계 카드 컴포넌트
│   └──  ZoneCanvas.jsx       # 영상 위 구역 그리기 캔버스
│
├── hooks/
│   ├── useApi.js             # 주기적 폴링 훅 (3초 간격)
│   └── useSSE.js             # SSE 실시간 알림 훅
│
├── pages/
│   ├── MainPage.jsx          # 메인 페이지 (영상 스트림, 알림 피드, 통계)
│   └── ViolationsPage.jsx    # 위반 기록 페이지 (목록, 필터, 상세 모달)
│
├── constants.js              # 위반 유형, 색상 등 공통 상수
├── utils.js                  # 날짜/시간 포맷 유틸
└── App.jsx                   # 라우팅 설정
```

---

## 페이지 설명

### 메인 페이지 (`/`)
- 영상 스트림 영역 (AI 서버 연결 후 활성화)
- 실시간 위반 알림 피드 (최근 6건)
- 오늘 위반 통계 카드 (총계 / 헬멧 미착용 / 인도 주행 / 다인 탑승)
- 새 위반 감지 시 토스트 알림 (SSE)

### 위반 기록 페이지 (`/violations`)
- 전체 위반 기록 테이블
- 위반 유형별 필터 버튼
- 상세 보기 모달 (캡처 이미지, 위반 유형, 감지 시각, 카메라, 신뢰도)

---

## 백엔드 API 연결

| 기능 | 방식 | 엔드포인트 |
|------|------|------------|
| 위반 기록 조회 | GET | `/api/violations?limit={n}` |
| 통계 조회 | GET | `/api/stats` |
| 실시간 알림 구독 | SSE | `/api/stream` |

---

## 데이터 흐름

```
백엔드 서버 (localhost:8080)
        │
        ├─ GET /api/violations  →  위반 기록 테이블 / 실시간 피드
        ├─ GET /api/stats       →  통계 카드
        └─ GET /api/stream      →  SSE 토스트 알림 (새 위반 감지 시)
```
