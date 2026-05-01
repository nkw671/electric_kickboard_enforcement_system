import cv2
#전역설정
SOURCE          = "src/1.mp4"
MODEL_PATH      = "src/best_v3.pt"
CONF            = 0.5
COOLDOWN        = 3.0
ZONE_FILE       = "zones.json"
CAMERA_ID       = "CAM-01"
VIOLATION_DIR   = "../violations"
BACKEND_URL     = "http://localhost:8080/api/violations"
ENCODE_PARAMS   = [cv2.IMWRITE_JPEG_QUALITY, 65]   # JPEG 인코딩 파라미터

latest_frame: bytes = b""   # MJPEG 스트림용 최신 프레임
alert_history: list = []    # 누적 알림 목록
