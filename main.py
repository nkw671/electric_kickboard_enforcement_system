import cv2
import json
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
from PIL import ImageFont, ImageDraw, Image

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ──────────────────────────────────────────────
# 설정값 및 경로 정의
# ──────────────────────────────────────────────
# 입력 영상, 모델 경로, 감지 임계값 등 시스템 운영에 필요한 기본 설정값들을 정의한다.
SOURCE = "src/1.mp4"
MODEL_PATH = "src/best_v3.pt"
CONF = 0.5
COOLDOWN = 3.0
ZONE_FILE = "zones.json"


class ZoneDrawer:
    """
    영상 프레임 위에 감시 구역(Zone)을 렌더링하고 관리하는 클래스.
    서버 환경에 맞춰 GUI 기능을 배제하고 데이터 처리와 그리기 기능만 수행한다.
    """

    # 구역별 구분을 위한 BGR 색상 팔레트 정의
    COLORS = [
        (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 100, 0), (255, 0, 200),
    ]

    # 함수 이름 : __init__()
    # 기능      : ZoneDrawer 객체를 초기화하고 폰트를 로드한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def __init__(self):
        self.zones = []  # 완성된 구역 목록 저장 변수
        self._zone_num = 1  # 구역 번호 관리 변수
        self._font = self._load_font(18)  # 한글 출력을 위한 폰트 객체 생성

    # 함수 이름 : _load_font()
    # 기능      : 운영체제별 경로를 탐색하여 사용 가능한 한글 폰트를 로드한다.
    # 파라미터  : int size -> 폰트 크기
    # 반환값    : ImageFont -> 로드된 폰트 객체
    @staticmethod
    def _load_font(size: int):
        # 시스템 환경에 따른 폰트 파일 후보 경로 목록
        candidates = [
            "malgun.ttf",
            "C:/Windows/Fonts/malgun.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)  # 폰트 로드 시도
            except OSError:
                continue  # 실패 시 다음 경로 시도
        return ImageFont.load_default()  # 모든 경로 실패 시 기본 폰트 반환

    # 함수 이름 : add_zone()
    # 기능      : 전달받은 좌표 리스트로 새로운 구역을 생성하고 저장한다.
    # 파라미터  : list pts -> 구역을 구성하는 [x, y] 좌표들의 리스트
    # 반환값    : str -> 생성된 구역의 이름
    def add_zone(self, pts: list):
        color = self.COLORS[self._zone_num % len(self.COLORS)]  # 팔레트에서 순차적으로 색상 선택
        name = f"Zone-{self._zone_num}"  # 구역 이름 생성
        self.zones.append({"name": name, "pts": pts, "color": color})  # 데이터 리스트에 추가
        self._zone_num += 1  # 다음 구역 번호 증가
        self.save(ZONE_FILE)  # 변경된 구역 정보를 파일로 즉시 저장
        return name

    # 함수 이름 : draw_zones()
    # 기능      : 현재 설정된 모든 구역을 프레임 위에 반투명하게 그린다.
    # 파라미터  : np.ndarray frame -> 그림을 그릴 대상 이미지 프레임
    # 반환값    : 없음
    def draw_zones(self, frame: np.ndarray):
        # 설정된 모든 구역을 순회하며 렌더링한다.
        for z in self.zones:
            poly = np.array(z["pts"], dtype=np.int32)  # 좌표 리스트를 넘파이 배열로 변환

            # 구역 내부 채우기 (반투명 효과 적용)
            ov = frame.copy()  # 오버레이용 복사본 생성
            cv2.fillPoly(ov, [poly], z["color"])  # 다각형 내부 색상 채우기
            cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)  # 원본과 합성하여 투명도 조절

            cv2.polylines(frame, [poly], True, z["color"], 2)  # 구역 외곽선 그리기

            # 구역 이름을 중심 좌표에 출력한다.
            cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())  # 중심점 계산
            cv2.putText(frame, z["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)  # 구역명 텍스트 출력

    # 함수 이름 : save()
    # 기능      : 현재 구역 목록을 JSON 파일로 직렬화하여 저장한다.
    # 파라미터  : str path -> 저장할 파일의 경로
    # 반환값    : 없음
    def save(self, path: str):
        # JSON 형식으로 데이터를 구성하여 파일에 쓴다.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 저장 시각 기록
                "zones": [{"name": z["name"], "pts": z["pts"], "color": list(z["color"])} for z in self.zones],
            }, f, indent=2)  # 가독성을 위해 들여쓰기 포함 저장
        print(f"[저장] {len(self.zones)}개 구역 정보가 {path}에 저장되었습니다.")

    # 함수 이름 : load()
    # 기능      : JSON 파일로부터 구역 정보를 읽어와 리스트에 복원한다.
    # 파라미터  : str path -> 불러올 파일의 경로
    # 반환값    : 없음
    def load(self, path: str):
        # 파일을 열어 데이터를 파싱하고 변수에 할당한다.
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.zones = [
            {"name": z["name"], "pts": [tuple(p) for p in z["pts"]], "color": tuple(z["color"])}
            for z in data["zones"]
        ]

        # 불러온 데이터가 있다면 구역 번호를 이어서 설정한다.
        if self.zones:
            self._zone_num = len(self.zones) + 1  # 마지막 번호 다음부터 시작
        print(f"[불러오기] {len(self.zones)}개의 구역을 로드했습니다.")

    # 함수 이름 : in_zone()
    # 기능      : 바운딩 박스의 하단 중앙 지점이 특정 구역 내부에 있는지 판별한다.
    # 파라미터  : dict zone -> 판별 대상 구역 정보
    #             int x1, y1 -> 박스 좌상단 좌표
    #             int x2, y2 -> 박스 우하단 좌표
    # 반환값    : bool -> 구역 내부 여부 (True/False)
    def in_zone(self, zone: dict, x1, y1, x2, y2) -> bool:
        poly = np.array(zone["pts"], dtype=np.int32)  # 구역 좌표 배열화
        cx, cy = int((x1 + x2) / 2), int(y2)  # 바운딩 박스의 하단 중앙(발 위치) 계산
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0  # 점의 다각형 내부 포함 여부 반환


# ──────────────────────────────────────────────
# 전역 객체 초기화 및 서버 설정
# ──────────────────────────────────────────────
# FastAPI 인스턴스 생성 및 AI 모델, 구역 관리 객체를 서버 시작 시점에 준비한다.
app = FastAPI()

# 웹 브라우저에서의 교차 출처 리소스 공유(CORS) 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[시스템] YOLO 모델 로딩 중...")
model = YOLO(MODEL_PATH).to('cuda')  # YOLOv8 객체 추적 모델 로드
drawer = ZoneDrawer()  # 구역 관리 객체 생성

# 기존 저장 파일이 존재할 경우 자동으로 데이터를 불러온다.
if os.path.exists(ZONE_FILE):
    drawer.load(ZONE_FILE)

last_alert = {}  # 알림 중복 방지를 위한 시각 저장소


# 함수 이름 : should_alert()
# 기능      : 특정 구역의 특정 객체에 대해 알림 쿨다운이 지났는지 확인한다.
# 파라미터  : str zone_name -> 대상 구역 이름
#             int tid -> 객체의 추적 ID
# 반환값    : bool -> 알림 발생 가능 여부
def should_alert(zone_name, tid):
    key = (zone_name, tid)  # 구역과 ID를 조합한 고유 키 생성
    # 현재 시각과 마지막 알림 시각의 차이를 계산한다.
    if time.time() - last_alert.get(key, 0) >= COOLDOWN:
        last_alert[key] = time.time()  # 쿨다운 경과 시 현재 시각 갱신
        return True  # 알림 가능
    return False  # 알림 대기


# ──────────────────────────────────────────────
# 영상 처리 및 송출 로직
# ──────────────────────────────────────────────

# 함수 이름 : generate_frames()
# 기능      : 영상을 한 프레임씩 읽어 AI 추적 및 구역 판별을 거친 후 JPEG 스트림으로 방출한다.
# 파라미터  : 없음
# 반환값    : bytes -> multipart 포맷의 JPEG 프레임 데이터
def generate_frames():
    cap = cv2.VideoCapture(SOURCE)  # 영상 파일 스트림 열기

    # 영상 재생이 끝날 때까지 무한 루프를 돌며 처리한다.
    while True:
        ret, frame = cap.read()  # 프레임 읽기 시도
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 재생 종료 시 처음으로 되돌리기
            continue

        # AI 모델을 사용하여 프레임 내 객체를 추적하고 결과를 가져온다.
        results = model.track(frame, persist=True, tracker='bytetrack.yaml',vid_stride=3 , conf=CONF, verbose=False)
        drawer.draw_zones(frame)  # 설정된 감시 구역 렌더링

        # 감지된 객체가 있을 경우 후처리를 진행한다.
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 박스 좌표 추출
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 클래스 ID 추출
            ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else range(
                len(boxes))

            ann = Annotator(frame, line_width=2)  # 박스 드로잉 도구 생성

            for box, cid, tid in zip(boxes, cls_ids, ids):
                x1, y1, x2, y2 = map(int, box)  # 좌표 정수화
                ann.box_label((x1, y1, x2, y2), f"{model.names[cid]} #{tid}", color=yolo_colors(cid, bgr=True))

                # 현재 객체가 설정된 어떤 구역이라도 침범했는지 확인한다.
                for z in drawer.zones:
                    if drawer.in_zone(z, x1, y1, x2, y2):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 침범 시 빨간 박스 표시

                        # 중복 알림 방지 로직을 거쳐 로그를 출력한다.
                        if should_alert(z["name"], int(tid)):
                            print(f"[경보] {datetime.now().strftime('%H:%M:%S')} | {z['name']} 침범 | ID: {tid}")

        # 처리된 프레임을 웹 송출을 위해 JPEG 형식으로 인코딩한다.
        ret_img, buffer = cv2.imencode('.jpg', frame)
        if ret_img:
            # 프레임을 바이트로 변환하여 HTTP 스트림 규격에 맞춰 전송한다.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()  # 자원 반납


# ──────────────────────────────────────────────
# FastAPI 엔드포인트(API) 정의
# ──────────────────────────────────────────────

# 프론트엔드로부터 좌표를 전달받기 위한 데이터 구조 클래스
class ZoneRequest(BaseModel):
    pts: List[List[int]]


# 함수 이름 : video_stream()
# 기능      : 후처리된 영상을 실시간 스트리밍으로 제공한다.
# 파라미터  : 없음
# 반환값    : StreamingResponse -> MJPEG 비디오 스트림
@app.get("/stream")
def video_stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# 함수 이름 : create_zone()
# 기능      : 새로운 감시 구역 좌표를 수신하여 시스템에 반영한다.
# 파라미터  : ZoneRequest request -> JSON 형태의 좌표 데이터
# 반환값    : dict -> 처리 결과 메시지
@app.post("/api/zones")
def create_zone(request: ZoneRequest):
    name = drawer.add_zone(request.pts)  # 구역 추가 로직 호출
    return {"message": f"{name} 구역이 등록되었습니다."}  # 결과 반환


# 함수 이름 : clear_zones()
# 기능      : 등록된 모든 구역 정보를 삭제하고 초기화한다.
# 파라미터  : 없음
# 반환값    : dict -> 처리 결과 메시지
@app.delete("/api/zones")
def clear_zones():
    drawer.zones.clear()  # 리스트 비우기
    drawer._zone_num = 1  # 번호 초기화
    drawer.save(ZONE_FILE)  # 초기화된 상태 저장
    return {"message": "모든 구역이 삭제되었습니다."}


# 서버 실행 메인 루틴
if __name__ == "__main__":
    import uvicorn

    # 지정된 호스트와 포트로 ASGI 서버를 가동한다.
    uvicorn.run(app, host="0.0.0.0", port=8000)