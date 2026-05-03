# AI Image Detection

킥보드 위반 행위를 실시간으로 감지하고 백엔드로 전송하는 AI 서버입니다.  
YOLO 객체 탐지 + ByteTrack 추적을 기반으로 헬멧 미착용 / 인도주행 / 다인탑승을 판정합니다.

---

## 실행 환경

- Python 3.10+
- CUDA 12.4 (PyTorch GPU 사용)
- 의존 패키지: `ultralytics`, `fastapi`, `uvicorn`, `opencv-python`, `pillow`, `httpx`, `numpy`

---

## 실행 방법

```
main.py 실행
```

---

## 파일 구조

```
AI/
├── main.py         # 진입점 — 객체 조립 및 서버·감지 루프 실행
├── config.py       # 전역 설정 상수 관리
├── zone.py         # Zone 데이터 관리(ZoneDrawer) + Zone 렌더링(ZoneRenderer)
├── violation.py    # 위반 전략 추상 클래스·3개 구현체 + 종합 판정
├── detection.py    # 영상 캡처·YOLO 추론·프레임 인코딩 루프(DetectionLoop)
├── api.py          # FastAPI 엔드포인트 + 스프링 부트 백엔드 위반 전송(ConnectAPI)
├── zones.json      # Zone 좌표 저장 파일 
├── train.py        # YOLO 모델 학습 스크립트
├── src/
│   ├── best_v3.pt      # 학습된 YOLO 모델 가중치
│   ├── 1.mp4           # 테스트용 영상 소스
│   └── servertest.py   # api 테스트용 Mock 서버 (추후 삭제 예정)
└── violations/
    └── *.jpg       # 위반 감지 캡처 이미지 (타임스탬프_유형_트래킹ID) (구현중으로 비활성)
```

---





## 위반 판정 기준

| 위반 유형 | 판정 기준 |
|---|---|
| 헬멧 미착용 | `helmet_X` 박스 중심이 탑승자 박스 내부에 있을 때 |
| 인도주행 | 탑승자 박스 하단 중심이 Zone 내부에 있을 때 |
| 다인탑승 | `2-person_with_kickboard` 레이블 감지 시 |

---

## 주요 설정 (config.py)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `SOURCE` | `src/1.mp4` | 영상 입력 경로 |
| `MODEL_PATH` | `src/best_v3.pt` | YOLO 모델 경로 |
| `CONF` | `0.5` | 감지 신뢰도 임계값 |
| `COOLDOWN` | `3.0` | 동일 객체 재알림 최소 간격 (초) |
| `CAMERA_ID` | `CAM-01` | 카메라 식별자 |
| `BACKEND_URL` | `http://localhost:8080/api/violations` | 위반 전송 대상 |

---

