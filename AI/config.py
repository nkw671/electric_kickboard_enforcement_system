import cv2
import os
#전역설정
SOURCE          = "src/1.mp4"
MODEL_PATH      = "src/kickboard+helmet+background.pt"
CONF            = 0.35
COOLDOWN        = 3.0
ZONE_FILE       = "zones.json"
CAMERA_ID       = "CAM-01"
VIOLATION_DIR   = "violations"
BACKEND_URL     = "http://localhost:8080/api/violations"
AI_BASE_URL     = os.environ.get("AI_BASE_URL", "http://localhost:8000")   # 배포 시 환경변수로 재정의
ENCODE_PARAMS      = [cv2.IMWRITE_JPEG_QUALITY, 65]   # JPEG 인코딩 파라미터
EMA_ALPHA          = 0.7   # EMA 스무딩 계수 (높을수록 현재 프레임 반영 비중 증가)
MIN_CONFIRM_FRAMES = 3     # 화면에 표시하기 위한 최소 연속 감지 프레임 수
INFER_SIZE      = (800, 450)                       # YOLO 추론용 축소 해상도 — 학습 imgsz=800 기준 (width, height)

latest_frame: bytes = b""   # MJPEG 스트림용 최신 프레임
alert_history: list = []    # 누적 알림 목록
