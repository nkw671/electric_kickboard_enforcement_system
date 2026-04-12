import cv2
import json
import asyncio
import numpy as np
import httpx
import os
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware



# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────

SOURCE          = "src/1.mp4"         # 입력 영상 경로 (웹캠은 0)
MODEL_PATH      = "src/best_v3.pt"  # YOLO 모델 가중치 경로
CONF            = 0.5                 # 객체 감지 최소 신뢰도
COOLDOWN        = 3.0                 # 동일 객체 재알림 최소 간격 (초)
ZONE_FILE       = "zones.json"        # Zone 좌표 저장/불러오기 파일 경로
CAMERA_ID       = "CAM-01"           # 카메라 식별자
VIOLATION_DIR   = "../violations"  # 위반 프레임 이미지 저장 디렉토리
BACKEND_URL     = "http://localhost:8080/api/violations"  # 위반 전송 백엔드 URL

# ──────────────────────────────────────────────
# 전역 공유 상태
# ──────────────────────────────────────────────
latest_frame: bytes = b""            # MJPEG 스트림용 최신 프레임
alert_history: list = []             # 누적 알림 목록 (GET /alerts 용)
drawer        = None                 # run_detection() 에서 초기화 후 ZoneAPI 와 공유


# ──────────────────────────────────────────────
# ZoneDrawer 클래스 (서버용)
#
# 내장 함수 목록:
#   __init__()     - Zone 상태 변수 초기화
#   _color()       - 현재 색상 인덱스에 해당하는 BGR 색상 튜플 반환
#   finish_zone()  - 현재 꼭짓점으로 Zone 완성하여 zones 목록에 추가
#   draw_zones()   - 완성된 모든 Zone을 반투명 채우기와 외곽선으로 프레임에 표시
#   _draw_dashed() - 두 점 사이를 일정 간격의 점선으로 표시
#   save()         - 현재 zones 목록을 JSON 파일로 저장
#   load()         - JSON 파일에서 Zone 목록을 불러와 zones에 저장
#   set_zones()    - API 에서 받은 좌표 목록으로 zones 를 교체한다
#
# ※ in_zone() 은 DecideViolation 으로 이동
# ──────────────────────────────────────────────
class ZoneDrawer:

    # Zone 색상 팔레트 (BGR 순서)
    COLORS = [
        (0,   0,   255),
        (0,  165,  255),
        (0,  255,    0),
        (255, 100,   0),
        (255,   0,  200),
    ]

    # 함수 이름 : __init__()
    # 기능      : Zone 상태 변수를 초기화한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def __init__(self):
        self.zones     = []   # 완성된 Zone 딕셔너리 목록
        self._pts      = []   # 현재 그리는 중인 꼭짓점 좌표 목록
        self._cidx     = 0    # 현재 선택된 색상 인덱스
        self._zone_num = 1    # 다음 Zone 이름에 붙을 번호

    # 함수 이름 : _color()
    # 기능      : 현재 색상 인덱스에 해당하는 BGR 색상 튜플을 반환한다.
    # 파라미터  : 없음
    # 반환값    : tuple -> (B, G, R)
    def _color(self):
        return self.COLORS[self._cidx % len(self.COLORS)]

    # 함수 이름 : finish_zone()
    # 기능      : 현재까지 찍은 꼭짓점으로 Zone을 완성하여 zones 목록에 추가한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def finish_zone(self):
        if len(self._pts) < 3:
            return
        name = f"Zone-{self._zone_num}"
        self.zones.append({"name": name, "pts": list(self._pts), "color": self._color()})
        self._pts      = []
        self._cidx    += 1
        self._zone_num += 1

    # 함수 이름 : draw_zones()
    # 기능      : zones 목록에 있는 모든 완성된 Zone을
    #             반투명 채우기와 외곽선으로 프레임에 그린다.
    # 파라미터  : np.ndarray frame -> 그림을 그릴 대상 프레임 (BGR 이미지)
    # 반환값    : 없음
    def draw_zones(self, frame: np.ndarray):
        for z in self.zones:
            poly = np.array(z["pts"], dtype=np.int32)

            # 반투명 채우기 : 원본 프레임과 채운 오버레이를 25:75 비율로 합성한다.
            ov = frame.copy()
            cv2.fillPoly(ov, [poly], z["color"])
            cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [poly], True, z["color"], 2)   # 외곽선 그리기

            # Zone 이름을 다각형 중심에 표시한다.
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, z["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)

    # 함수 이름 : _draw_dashed()
    # 기능      : 두 점 사이를 일정 간격의 점선으로 그린다.
    # 파라미터  : np.ndarray img -> 그릴 대상 이미지
    #             tuple      p1  -> 시작점 (x, y)
    #             tuple      p2  -> 끝점   (x, y)
    #             tuple   color  -> 선 색상 (B, G, R)
    #             int       gap  -> 점선 간격 (픽셀, 기본값 8)
    # 반환값    : 없음
    def _draw_dashed(self, img, p1, p2, color, gap=8):
        x1, y1 = p1
        x2, y2 = p2
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if dist == 0:
            return
        n = max(int(dist / gap), 1)
        for i in range(0, n, 2):
            t1, t2 = i / n, min((i + 1) / n, 1.0)
            cv2.line(img,
                     (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1)),
                     (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2)),
                     color, 1, cv2.LINE_AA)

    # 함수 이름 : save()
    # 기능      : 현재 zones 목록을 JSON 파일로 저장한다.
    # 파라미터  : str path -> 저장할 파일 경로
    # 반환값    : 없음
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "zones": [
                        {"name": z["name"], "pts": z["pts"], "color": list(z["color"])}
                        for z in self.zones
                    ],
                },
                f, indent=2,
            )
        print(f"[저장] {len(self.zones)}개 Zone -> {path}")

    # 함수 이름 : load()
    # 기능      : JSON 파일에서 Zone 목록을 불러와 zones 에 저장한다.
    # 파라미터  : str path -> 불러올 파일 경로
    # 반환값    : 없음
    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.zones = [
            {
                "name":  z["name"],
                "pts":   [tuple(p) for p in z["pts"]],
                "color": tuple(z["color"]),
            }
            for z in data["zones"]
        ]
        print(f"[불러오기] {len(self.zones)}개 Zone <- {path}")

    # 함수 이름 : set_zones()
    # 기능      : React 캔버스에서 그린 Zone 좌표를 API 로 받아 zones 를 교체한다.
    #             color 가 없으면 팔레트에서 자동 배정하고 파일에 저장한다.
    # 파라미터  : list zone_list -> [{"name": str, "pts": [[x,y],...], "color"(선택): [B,G,R]}, ...]
    # 반환값    : 없음
    def set_zones(self, zone_list: list):
        self.zones = []
        for i, z in enumerate(zone_list):
            color = tuple(z["color"]) if "color" in z else self.COLORS[i % len(self.COLORS)]
            self.zones.append({
                "name":  z.get("name", f"Zone-{i + 1}"),
                "pts":   [tuple(p) for p in z["pts"]],
                "color": color,
            })
        self.save(ZONE_FILE)


# ──────────────────────────────────────────────
# DecideViolation 클래스
# 매 프레임에서 위반 항목을 판정하고 백엔드로 전송하는 클래스.
#
# 내장 함수 목록:
#   __init__()              - 모델, ZoneDrawer, 쿨다운, 콜백 초기화
#   in_zone()               - 바운딩 박스 하단 중심이 특정 Zone 내부인지 확인 (ZoneDrawer에서 이동)
#   _is_helmet_missing()    - helmet_x 가 person_with_kickboard / 2+person_with_kickboard 안에 있으면 헬멧 미착용
#   _is_sidewalk_riding()   - person_with_kickboard / 2+person_with_kickboard 가 Zone 안에 있으면 인도주행
#   _is_double_riding()     - 2+person_with_kickboard 레이블이 감지되면 다인탑승
#   _save_frame()           - 위반 발생 시 해당 프레임을 이미지 파일로 저장
#   _should_alert()         - 동일 객체·위반 유형에 대한 쿨다운 여부 확인
#   check()                 - 매 프레임 위반 항목을 종합 판정하고 콜백을 호출
#   run()                   - 영상 캡처 및 YOLO 추적 루프 실행 (run_detection에서 이동)
# ──────────────────────────────────────────────
class DecideViolation:

    # 판정 대상 레이블
    RIDER_LABELS = {"person_with_kickboard", "2-person_with_kickboard"}
    HELMET_LABEL = "helmet_X"
    DOUBLE_LABEL = "2-person_with_kickboard"                             # 다인탑승 레이블

    # 함수 이름 : __init__()
    # 기능      : DecideViolation 객체를 초기화한다.
    #             YOLO 모델, ZoneDrawer, 쿨다운 딕셔너리, 위반 콜백을 설정한다.
    # 파라미터  : YOLO        model       -> 이미 로드된 YOLO 모델 인스턴스
    #             ZoneDrawer  zone_drawer -> 인도 Zone 정보를 가진 ZoneDrawer 인스턴스
    #             callable    on_violation -> 위반 감지 시 호출할 콜백 함수
    #                                        인자: (violation_type: str, track_id: int, conf: float, frame: np.ndarray)
    # 반환값    : 없음
    def __init__(self, model: YOLO, zone_drawer: "ZoneDrawer", on_violation: callable):
        self.model        = model
        self.zone_drawer  = zone_drawer
        self.on_violation = on_violation   # 위반 감지 시 ZoneAPI.send_violation() 호출
        self._last_alert  = {}             # 쿨다운 관리 딕셔너리 {(track_id, violation_type): timestamp}
        self._font = self._load_font(20)  # 한글 폰트 로드

    @staticmethod
    def _load_font(size: int):
        candidates = [
            "malgunbd.ttf",
            "C:/Windows/Fonts/malgun.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        for path in candidates:
            try:
                from PIL import ImageFont
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        from PIL import ImageFont
        return ImageFont.load_default()

    def _put_korean_text(self, frame, text, pos, color=(0, 0, 255)):
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=self._font, fill=(color[2], color[1], color[0]))  # BGR→RGB
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        np.copyto(frame, result)

    # 함수 이름 : in_zone()
    # 기능      : 바운딩 박스의 하단 중심(발 위치)이 특정 Zone 내부인지 확인한다.
    #             ZoneDrawer 에서 이동.
    # 파라미터  : dict zone       -> 확인할 Zone 딕셔너리
    #             int  x1, y1    -> 바운딩 박스 좌상단 좌표 (픽셀)
    #             int  x2, y2    -> 바운딩 박스 우하단 좌표 (픽셀)
    # 반환값    : bool -> True 이면 발 위치가 Zone 내부
    def in_zone(self, zone: dict, x1: int, y1: int, x2: int, y2: int) -> bool:
        poly   = np.array(zone["pts"], dtype=np.int32)
        cx, cy = int((x1 + x2) / 2), int(y2)   # 발 위치 좌표
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

    # 함수 이름 : _is_helmet_missing()
    # 기능      : helmet_x 바운딩 박스가 person_with_kickboard 또는
    #             2+person_with_kickboard 바운딩 박스 내부에 포함되면 헬멧 미착용으로 판정한다.
    # 파라미터  : list boxes    -> 전체 바운딩 박스 배열 [(x1,y1,x2,y2), ...]
    #             list labels   -> 각 박스에 대응하는 레이블 문자열 목록
    #             int  rider_idx -> 탑승자 박스의 인덱스
    # 반환값    : bool -> True 이면 헬멧 미착용
    def _is_helmet_missing(self, boxes: list, labels: list, rider_idx: int) -> bool:
        rx1, ry1, rx2, ry2 = boxes[rider_idx]   # 탑승자 바운딩 박스 좌표

        for i, label in enumerate(labels):
            if label != self.HELMET_LABEL:
                continue

            # helmet_x 박스의 중심이 탑승자 박스 내부에 있는지 확인한다.
            hx1, hy1, hx2, hy2 = boxes[i]
            hcx = (hx1 + hx2) / 2   # helmet_x 박스 중심 x
            hcy = (hy1 + hy2) / 2   # helmet_x 박스 중심 y

            if rx1 <= hcx <= rx2 and ry1 <= hcy <= ry2:
                return True          # 탑승자 박스 안에 helmet_x 존재 → 헬멧 미착용

        return False

    # 함수 이름 : _is_sidewalk_riding()
    # 기능      : 탑승자 바운딩 박스 하단 중심이 설정된 Zone 중 하나라도 내부에 있으면
    #             인도주행으로 판정한다.
    # 파라미터  : int x1, y1, x2, y2 -> 탑승자 바운딩 박스 좌표
    # 반환값    : bool -> True 이면 인도주행
    def _is_sidewalk_riding(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        for zone in self.zone_drawer.zones:
            if self.in_zone(zone, x1, y1, x2, y2):
                return True
        return False

    # 함수 이름 : _is_double_riding()
    # 기능      : 감지된 레이블 중 2+person_with_kickboard 가 있으면 다인탑승으로 판정한다.
    # 파라미터  : str label -> 현재 객체의 레이블 문자열
    # 반환값    : bool -> True 이면 다인탑승
    def _is_double_riding(self, label: str) -> bool:
        return label == self.DOUBLE_LABEL

    # 함수 이름 : _save_frame()
    # 기능      : 위반이 발생한 프레임을 이미지 파일로 저장하고 로컬 경로를 반환한다.
    # 파라미터  : np.ndarray frame          -> 저장할 프레임 (BGR 이미지)
    #             str        violation_type -> 위반 유형 문자열 (파일명에 포함)
    #             int        track_id       -> 추적 ID (파일명에 포함)
    # 반환값    : str -> 저장된 이미지의 로컬 파일 경로
    def _save_frame(self, frame: np.ndarray, violation_type: str, track_id: int) -> str:
        os.makedirs(VIOLATION_DIR, exist_ok=True)   # 저장 디렉토리가 없으면 생성한다.

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{violation_type}_{track_id}.jpg"
        filepath  = os.path.join(VIOLATION_DIR, filename)

        cv2.imwrite(filepath, frame)                # 프레임을 JPEG 로 저장한다.
        return filepath

    # 함수 이름 : _should_alert()
    # 기능      : 동일 객체·위반 유형 조합에 대해 쿨다운이 지났는지 확인한다.
    #             쿨다운 내에 있으면 중복 전송을 방지한다.
    # 파라미터  : int track_id       -> 객체 추적 ID
    #             str violation_type -> 위반 유형 문자열
    # 반환값    : bool -> True 이면 알림 전송 가능
    def _should_alert(self, track_id: int, violation_type: str) -> bool:
        import time
        key = (track_id, violation_type)
        if time.time() - self._last_alert.get(key, 0) >= COOLDOWN:
            self._last_alert[key] = time.time()   # 마지막 알림 시각 갱신
            return True
        return False

    # 함수 이름 : check()
    # 기능      : 한 프레임의 YOLO 감지 결과를 받아 위반 항목을 종합 판정한다.
    #             위반이 감지되고 쿨다운이 지났으면 on_violation 콜백을 호출한다.
    # 파라미터  : np.ndarray      frame   -> 현재 처리 중인 영상 프레임
    #             list            boxes   -> 바운딩 박스 좌표 목록 [(x1,y1,x2,y2), ...]
    #             list            labels  -> 각 박스에 대응하는 레이블 문자열 목록
    #             list            confs   -> 각 박스에 대응하는 신뢰도 목록
    #             list            ids     -> 각 박스에 대응하는 추적 ID 목록
    # 반환값    : 없음
    def check(self, frame: np.ndarray, boxes: list, labels: list,
              confs: list, ids: list):
        for i, label in enumerate(labels):
            if label not in self.RIDER_LABELS:   # 탑승자 레이블이 아니면 건너뜀
                continue

            x1, y1, x2, y2 = boxes[i]
            tid  = ids[i]
            conf = confs[i]

            # 위반 항목별로 판정하고 콜백을 호출한다.
            violations = []

            if self._is_helmet_missing(boxes, labels, i):
                violations.append("헬멧 미착용")

            if self._is_sidewalk_riding(x1, y1, x2, y2):
                violations.append("인도주행")

            if self._is_double_riding(label):
                violations.append("다인탑승")

            for idx, v_type in enumerate(violations):
                self._put_korean_text(
                    frame, v_type,
                    pos=(x1, y1 - 60 - idx * 24),
                )
                if self._should_alert(tid, v_type):
                    # 위반 프레임을 저장한다.
                    saved_path = self._save_frame(frame, v_type, tid) #잠깐 비활성

                    # 콜백(ZoneAPI.send_violation)을 호출한다.
                    self.on_violation(
                        violation_type = v_type,
                        track_id       = tid,
                        conf           = conf,
                        # image_path   = saved_path,  # 추후 image_url 연동 시 활성화
                    )
                    print(f"[VIOLATION] {v_type} | #{tid} | conf={conf:.2f}")

    # 함수 이름 : run()
    # 기능      : 영상을 프레임 단위로 읽고 YOLO 로 추적하면서 check() 를 호출한다.
    #             처리된 프레임을 JPEG 로 인코딩하여 latest_frame 에 저장한다.
    #             run_detection() 에서 이동.
    # 파라미터  : 없음
    # 반환값    : 없음
    def run(self):
        global latest_frame

        cap = cv2.VideoCapture(SOURCE)
        print("\n[감지 루프 시작]\n")

        while True:
            ret, frame = cap.read()
            if not ret:                             # 영상 끝에 도달하면 처음부터 재생한다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # YOLO 로 객체를 추적한다.
            results = self.model.track(
                frame, persist=True, conf=CONF, verbose=False,
                tracker="bytetrack.yaml", vid_stride=2
            )

            self.zone_drawer.draw_zones(frame)      # Zone 오버레이 렌더링

            if results[0].boxes is not None:
                raw_boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
                confs     = results[0].boxes.conf.cpu().numpy().tolist()
                ids       = (
                    results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    if results[0].boxes.id is not None
                    else list(range(len(raw_boxes)))
                )
                labels = [self.model.names[c] for c in cls_ids]
                boxes  = [tuple(map(int, b)) for b in raw_boxes]

                ann = Annotator(frame, line_width=2)
                for box, label, conf, tid in zip(boxes, labels, confs, ids):
                    ann.box_label(box, f"{label} #{tid} {conf:.2f}",
                                  color=yolo_colors(cls_ids[boxes.index(box)], bgr=True))

                # 위반 판정을 수행한다.
                self.check(frame, boxes, labels, confs, ids)

            # 처리된 프레임을 JPEG 로 인코딩하여 스트리밍용 전역 변수에 저장한다.
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            latest_frame = buf.tobytes()

        cap.release()


# ──────────────────────────────────────────────
# ZoneAPI 클래스
# FastAPI 라우터와 엔드포인트를 하나로 묶는다.
#
# 내장 함수 목록:
#   __init__()          - FastAPI 앱, CORS, 라우터를 초기화하고 엔드포인트를 등록한다
#   send_violation()    - 위반 정보를 백엔드 POST /api/violations 로 전송한다
#   video_stream()      - latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다
#   get_zones()         - 현재 설정된 Zone 목록을 반환한다
#   set_zones()         - React 캔버스에서 그린 Zone 목록을 받아 저장한다
#   delete_zones()      - 모든 Zone 을 초기화하고 파일을 갱신한다
#   get_alerts()        - 누적된 침범 알림 목록을 반환한다
# ──────────────────────────────────────────────
class ConnectAPI:

    # 함수 이름 : __init__()
    # 기능      : FastAPI 앱을 생성하고 CORS 설정 및 엔드포인트를 등록한다.
    # 파라미터  : ZoneDrawer drawer        -> 감지 루프와 공유하는 ZoneDrawer 인스턴스
    #             list       alert_history -> 감지 루프와 공유하는 알림 목록
    # 반환값    : 없음
    def __init__(self, drawer: "ZoneDrawer", alert_history: list):
        self.drawer        = drawer
        self.alert_history = alert_history

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 엔드포인트를 등록한다.
        self.app.get("/video/stream")(self.video_stream)
        self.app.get("/zones")(self.get_zones)
        self.app.post("/zones")(self.set_zones)
        self.app.delete("/zones")(self.delete_zones)
        self.app.get("/alerts")(self.get_alerts)

    # 함수 이름 : send_violation()
    # 기능      : 위반 정보를 백엔드 POST /api/violations 로 전송하고
    #             alert_history 에도 추가한다.
    #             DecideViolation.check() 의 on_violation 콜백으로 호출된다.
    # 파라미터  : str   violation_type -> 위반 유형 ("헬멧 미착용" / "인도주행" / "다인탑승")
    #             int   track_id       -> 객체 추적 ID
    #             float conf           -> YOLO 감지 신뢰도 (0.0 ~ 1.0)
    # 반환값    : 없음
    def send_violation(self, violation_type: str, track_id: int, conf: float):
        payload = {
            "type": violation_type,
            # "image_url": image_url,
            "camera": CAMERA_ID,
            "confidence": int(conf * 100),
        }

        alert = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": violation_type,
            "track_id": track_id,
            "confidence": payload["confidence"],
            "camera": CAMERA_ID,
        }
        self.alert_history.append(alert)
        print(alert)

        # 전송을 별도 스레드에서 실행하여 감지 루프를 블로킹하지 않는다.
        Thread(target=self._post_violation, args=(payload,), daemon=True).start()

    # 함수 이름 : _post_violation()
    # 기능      : 백엔드로 위반 정보를 전송한다.
    #             send_violation() 에서 별도 스레드로 호출된다.
    # 파라미터  : dict payload -> 전송할 위반 정보 딕셔너리
    # 반환값    : 없음
    def _post_violation(self, payload: dict):
        try:
            with httpx.Client() as client:
                client.post(BACKEND_URL, json=payload, timeout=3.0)
        except Exception as e:
            print(f"[전송 실패] {payload.get('type')} | {e}")

    # 함수 이름 : video_stream()
    # 기능      : latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다.
    # 파라미터  : 없음
    # 반환값    : StreamingResponse (multipart/x-mixed-replace)
    async def video_stream(self):
        async def generate():
            while True:
                if latest_frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + latest_frame +
                        b"\r\n"
                    )
                await asyncio.sleep(0.03)   # 약 30fps
        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # 함수 이름 : get_zones()
    # 기능      : 현재 설정된 Zone 목록을 반환한다.
    # 파라미터  : 없음
    # 반환값    : {"zones": [...]}
    async def get_zones(self):
        return {
            "zones": [
                {"name": z["name"], "pts": z["pts"], "color": list(z["color"])}
                for z in self.drawer.zones
            ]
        }

    # 함수 이름 : set_zones()
    # 기능      : React 캔버스에서 그린 Zone 목록을 받아 drawer 에 저장한다.
    # 파라미터  : body = {"zones": [{"name": str, "pts": [[x,y],...], "color": [B,G,R]}, ...]}
    # 반환값    : {"saved": Zone 개수}
    async def set_zones(self, body: dict):
        new_zones = []
        for i, z in enumerate(body.get("zones", [])):
            color = tuple(z["color"]) if "color" in z else self.drawer.COLORS[i % len(self.drawer.COLORS)]
            new_zones.append({
                "name":  z.get("name", f"Zone-{i + 1}"),
                "pts":   [tuple(p) for p in z["pts"]],
                "color": color,
            })
        self.drawer.zones = new_zones
        self.drawer.save(ZONE_FILE)
        return {"saved": len(self.drawer.zones)}

    # 함수 이름 : delete_zones()
    # 기능      : 모든 Zone 을 초기화하고 파일을 갱신한다.
    # 파라미터  : 없음
    # 반환값    : {"cleared": true}
    async def delete_zones(self):
        self.drawer.zones = []
        self.drawer.save(ZONE_FILE)
        return {"cleared": True}

    # 함수 이름 : get_alerts()
    # 기능      : 누적된 위반 알림 목록을 반환한다.
    #             after 파라미터로 특정 인덱스 이후 알림만 조회할 수 있다.
    # 파라미터  : int after -> 이 인덱스 이후의 알림만 반환 (기본값 0)
    # 반환값    : {"alerts": [...], "total": int}
    async def get_alerts(self, after: int = 0):
        return {
            "alerts": self.alert_history[after:],
            "total":  len(self.alert_history),
        }


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # YOLO 모델을 로드한다.
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully!")

    # ZoneDrawer 를 먼저 생성하고 저장된 Zone 을 불러온다.
    drawer = ZoneDrawer()
    if os.path.exists(ZONE_FILE):
        drawer.load(ZONE_FILE)

    # ZoneAPI 를 생성한다. send_violation 을 DecideViolation 의 콜백으로 전달한다.
    api = ConnectAPI(drawer=drawer, alert_history=alert_history)

    # DecideViolation 을 생성하고 on_violation 콜백에 api.send_violation 을 연결한다.
    detector = DecideViolation(
        model        = model,
        zone_drawer  = drawer,
        on_violation = api.send_violation,
    )

    # 감지 루프를 별도 스레드에서 실행한다.
    Thread(target=detector.run, daemon=True).start()

    # FastAPI 서버를 실행한다.
    uvicorn.run(api.app, host="0.0.0.0", port=8000)
