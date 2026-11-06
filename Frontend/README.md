# Frontend — KickDan 관제 대시보드

AI가 감지한 전동킥보드 위반을 실시간으로 확인하고, 단속 기록·통계를 조회하고, 인도 구역을 설정하는 대시보드입니다.

- 배포: https://kickdan.com (Vercel, 접근 코드 필요)
- 데모 모드: https://kickdan.com/?demo=1 (서버 없이 예시 데이터로 동작)

---

## 기술 스택

| 항목 | 기술 |
|------|------|
| 프레임워크 | React 19 |
| 빌드 도구 | Vite |
| 라우팅 | React Router 7 |
| 차트 | Recharts 3 |
| 스타일 | CSS Modules, Pretendard |
| 통신 | Fetch API(3초 폴링), SSE(새 위반 알림), MJPEG(탐지 영상) |
| 배포 | Vercel |

---

## 로컬 실행

```bash
cd Frontend
npm install     # 최초 1회
npm run dev     # http://localhost:5173
```

| 명령 | 설명 |
|---|---|
| `npm run dev` | 개발 서버 |
| `npm run build` | 프로덕션 빌드 (`dist/`) |
| `npm run preview` | 빌드 결과 미리보기 |
| `npm run lint` | ESLint |

### 서버 연결

로컬에서는 Vite 프록시가 요청을 넘겨줍니다(`vite.config.js`). 페이지 코드는 상대 경로로 요청합니다.

| 요청 경로 | 넘겨주는 곳 |
|---|---|
| `/api/*` | 백엔드 `http://localhost:8080` (경로 그대로) |
| `/ai/*` | AI 서버 `http://localhost:8000` (`/ai` 접두어를 떼고) |

- 백엔드만 켜도 모니터링의 최근 위반·알림, 단속 기록, 통계는 동작합니다.
- 탐지 영상과 인도 구역 설정은 AI 서버가 필요합니다.
- 서버를 하나도 켜지 않았다면 http://localhost:5173/?demo=1 로 접속합니다.

### 환경 변수

`.env.example`을 `.env`로 복사해서 씁니다. 모두 비워 두면 로컬 프록시를 쓰고 접근 코드 화면도 나오지 않습니다.

| 변수 | 설명 | 배포 값 |
|---|---|---|
| `VITE_API_BASE` | 백엔드 주소. 비우면 Vite 프록시 | `https://api.kickdan.com` |
| `VITE_AI_BASE` | AI 서버 주소. 비우면 Vite 프록시(`/ai`) | `https://ai.kickdan.com` |
| `VITE_ACCESS_CODE` | 접속할 때 입력하는 접근 코드. 비우면 접근 코드 화면 없음 | Vercel 프로젝트 설정에만 등록 |
| `VITE_DEMO_MODE` | `true`면 항상 데모 모드 | 비움 |

> `VITE_` 변수는 빌드 결과(JS)에 그대로 들어갑니다. 접근 코드는 발표용 간단한 잠금이며 보안 수단이 아닙니다.

---

## 데모 모드

AI 서버나 백엔드가 꺼져 있어도 화면을 보여줄 수 있도록 만든 발표 백업 기능입니다.

- 켜기: 주소에 `?demo=1` (탭을 닫을 때까지 유지) 또는 `VITE_DEMO_MODE=true`로 빌드
- 끄기: 주소에 `?demo=0`
- 켜지면 사이드바 아래에 "데모 모드" 배지가 표시됩니다.

`src/demo/installDemo.js`가 `fetch`와 `EventSource`를 가로채서 백엔드와 같은 형태로 응답합니다. 그래서 페이지 코드는 데모 여부를 몰라도 됩니다. 데모 코드는 데모 모드일 때만 불러오므로 평소 번들에는 들어가지 않습니다.

| 요청 | 데모 응답 |
|---|---|
| `GET /api/violations` | 최근 7일 위반 240건 (새로고침해도 같은 데이터) |
| `GET /api/stats` | 위 데이터로 계산한 통계 |
| `SSE /api/stream` | 10~20초마다 새 위반 |
| `/zones` | 브라우저 메모리에만 저장 |
| 탐지 영상 | `public/demo-stream.svg` |

---

## 페이지

| 경로 | 페이지 | 내용 |
|---|---|---|
| `/` | 모니터링 | 탐지 영상(전체화면 가능), 최근 위반 목록, 새 위반 알림(토스트). 목록·알림을 누르면 캡처 이미지 상세 창 |
| `/violations` | 단속 기록 | 날짜·기간·위반 유형 필터, 10건씩 페이지 이동, 상세 창 |
| `/stats` | 단속 통계 | 유형별 요약, 유형 분포(도넛), 인도 주행 많은 구역 TOP 5, 시간대별 현황(유형별 누적 막대) |
| `/zones` | 인도 구역 설정 | 영상 위에 다각형으로 인도 구역을 그리고 이름을 붙여 AI 서버에 저장 |

---

## 폴더 구조

```
src/
├── main.jsx                     # 진입점 (데모 모드 설치, 접근 코드 화면)
├── App.jsx                      # 라우팅, 페이지별 ErrorBoundary, 통계 페이지 지연 로딩
├── config.js                    # apiUrl() / aiUrl() — 환경 변수로 서버 주소 결정
├── constants.js                 # 위반 유형별 색상
├── typeIcons.jsx                # 위반 유형별 아이콘
├── utils.js                     # 응답 변환·날짜 유틸
│
├── components/
│   ├── Layout.jsx               # 사이드바 + 본문
│   ├── ViolationDetailModal.jsx # 위반 상세 창 (캡처 이미지, 이미지 실패 시 안내)
│   ├── AccessGate.jsx           # 접근 코드 입력 화면
│   └── ErrorBoundary.jsx        # 화면 오류 시 대체 화면
│
├── hooks/
│   ├── useApi.js                # 주기적 GET 폴링 (기본 3초)
│   └── useSSE.js                # SSE 구독 + 끊기면 자동 재연결
│
├── demo/
│   ├── demoMode.js              # 데모 모드 여부 판단
│   └── installDemo.js           # fetch/EventSource 가로채기
│
└── pages/                       # 페이지별 *.module.css 포함
    ├── MainPage.jsx
    ├── ViolationsPage.jsx
    ├── StatsPage.jsx
    └── ZonePage.jsx
```

새 위반 유형을 추가하면 `constants.js`의 색상 표와 `typeIcons.jsx`, `StatsPage.jsx`의 `TYPE_CONFIG`를 함께 수정해야 합니다.

---

## 사용하는 API

| 서버 | 방식 | 경로 | 사용하는 곳 |
|---|---|---|---|
| 백엔드 | GET | `/api/violations?size=n` | 모니터링(최근 10건), 단속 기록·통계(전체) |
| 백엔드 | GET | `/api/stats` | 통계 요약·도넛·구역 TOP 5 |
| 백엔드 | SSE | `/api/stream` | 모니터링 새 위반 알림 |
| AI | GET | `/video/stream` | 탐지 영상 (MJPEG, `<img>`로 표시) |
| AI | GET / POST / DELETE | `/zones` | 인도 구역 조회 / 전체 교체 저장 / 전체 삭제 |
| AI | GET | `/violations/*.jpg` | 위반 캡처 이미지 (기록의 `imageUrl`) |

---

## 구현하면서 정한 것들

- **필터와 페이지는 브라우저에서 처리**: 백엔드는 `page`/`size`/`type` 페이지네이션을 지원하지만, 날짜 조건과 유형별 시간대 집계가 없습니다. 시연 데이터가 수십~수백 건이라 기록 전체를 받아 날짜·유형 필터, 페이지 나누기, 시간대별 집계를 브라우저에서 합니다. 데이터가 많아지면 서버로 넘기도록 바꿀 곳을 코드 주석에 남겨 두었습니다.
- **부분 실패 처리**: API 하나가 실패하면 화면 전체가 아니라 그 데이터를 쓰는 카드에만 오류를 표시합니다. 이전 데이터가 있으면 폴링이 한 번 실패해도 그대로 보여줍니다.
- **자동 재연결**: SSE는 끊기면 점점 간격을 늘려 다시 연결하고, 탐지 영상과 인도 구역 화면도 AI 서버가 다시 켜지면 자동으로 이어집니다.
- **폴링 중 차트 깜빡임 방지**: 3초마다 새 응답 객체가 와도 값이 바뀔 때만 차트 데이터를 교체합니다. 다시 그릴 때마다 애니메이션을 처음부터 시작하는 도넛 차트는 애니메이션을 껐습니다.
- **첫 화면 로딩**: 차트 라이브러리가 큰 통계 페이지는 지연 로딩해서 첫 번들 크기를 줄였습니다.

---

## 배포 (Vercel)

- Vercel 프로젝트 `kickdan`에 `Frontend` 폴더를 연결 (`Frontend/.vercel`, gitignore됨)
- 환경 변수 `VITE_API_BASE`, `VITE_AI_BASE`, `VITE_ACCESS_CODE`는 Vercel 프로젝트 설정(Production)에 등록
- `vercel.json`이 모든 경로를 `index.html`로 보내서 `/stats` 같은 주소로 바로 들어가거나 새로고침해도 동작
- Git 연동 자동 배포가 아니라 CLI로 배포합니다

```bash
vercel deploy --prod --cwd Frontend
```

---

## 사용한 외부 자료

- 로고 번개 아이콘: "Thunder" by khaerul ilyas, [Noun Project](https://thenounproject.com/) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- 글꼴: [Pretendard](https://github.com/orioncactus/pretendard), [Plus Jakarta Sans](https://fonts.google.com/specimen/Plus+Jakarta+Sans) (SIL Open Font License)
