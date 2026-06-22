import cv2
import os
#전역설정
SOURCE          = "src/lowAngle_60fps.mp4"
MODEL_PATH      = "src/final_s.pt"
CONF            = 0.2    # 낮은 신뢰도 검출까지 ByteTrack 에 전달 → 2단계 연관으로 깜박임 방지
HELMET_CONF     = 0.4    # helmet_O / helmet_X 전용 최소 신뢰도
COOLDOWN         = 3.0
SUSTAIN          = 1.5   # 위반이 단속으로 인정되기 위한 최소 연속 지속 시간 (초)
TRACK_BUFFER_SEC = 5.0   # 객체 미감지 시 트랙 유지 시간 (초) — FPS에 따라 자동 환산
ZONE_FILE       = "zones.json"
CAMERA_ID       = "CAM-01"
VIOLATION_DIR   = "violations"
BACKEND_URL     = "http://localhost:8080/api/violations"
AI_BASE_URL     = os.environ.get("AI_BASE_URL", "http://localhost:8000")   # 배포 시 환경변수로 재정의
ENCODE_PARAMS      = [cv2.IMWRITE_JPEG_QUALITY, 65]   # JPEG 인코딩 파라미터
EMA_ALPHA          = 0.9  # EMA 스무딩 계수 (높을수록 현재 프레임 반영 비중 증가)
MIN_CONFIRM_FRAMES = 4     # 화면에 표시하기 위한 최소 연속 감지 프레임 수
COAST_MAX_FRAMES   = 1     # 검출이 빠진 확정 트랙의 마지막 박스를 유지할 최대 프레임 수 (coasting)
INFER_SIZE      = (960, 544)                       # YOLO 추론용 축소 해상도 — 32 배수 (width, height)
INFER_IMGSZ     = (INFER_SIZE[1], INFER_SIZE[0])   # model.track 에 넘길 imgsz (height, width) — INFER_SIZE 와 연동

latest_frame: bytes = b""   # MJPEG 스트림용 최신 프레임
alert_history: list = []    # 누적 알림 목록
