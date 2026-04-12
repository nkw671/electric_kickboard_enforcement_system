import cv2
import json
import asyncio
import time
import numpy as np
import httpx
import os
from datetime import datetime
from threading import Thread
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
SOURCE          = "src/1.mp4"
MODEL_PATH      = "src/best_v3.pt"
CONF            = 0.5
COOLDOWN        = 3.0
ZONE_FILE       = "zones.json"
CAMERA_ID       = "CAM-01"
VIOLATION_DIR   = "../violations"
BACKEND_URL     = "http://localhost:8080/api/violations"

# JPEG 인코딩 파라미터 — 한 번만 생성하여 재사용한다.
ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 65]

# ──────────────────────────────────────────────
# 전역 공유 상태
# ──────────────────────────────────────────────
latest_frame: bytes = b""
alert_history: list = []
drawer        = None


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
        self.zones     = []
        self._pts      = []
        self._cidx     = 0
        self._zone_num = 1

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
            cv2.polylines(frame, [poly], True, z["color"], 2)

            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, z["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)

    # 함수 이름 : _draw_dashed()
    # 기능      : 두 점 사이를 일정 간격의 점선으로 그린다.
    # 파라미터  : np.ndarray img, tuple p1, tuple p2, tuple color, int gap
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
#   __init__()              - 모델, ZoneDrawer, 쿨다운, 폰트, 콜백 초기화
#   _load_font()            - 시스템 한글 폰트 탐색 및 로드
#   _draw_violations()      - 한 프레임의 위반 텍스트를 PIL 변환 1회로 일괄 렌더링 (최적화)
#   in_zone()               - 바운딩 박스 하단 중심이 특정 Zone 내부인지 확인
#   _is_helmet_missing()    - helmet_X 가 탑승자 박스 안에 있으면 헬멧 미착용 판정
#   _is_sidewalk_riding()   - 탑승자가 Zone 안에 있으면 인도주행 판정
#   _is_double_riding()     - 2-person_with_kickboard 레이블이면 다인탑승 판정
#   _save_frame()           - 위반 프레임을 이미지 파일로 저장
#   _should_alert()         - 동일 객체·위반 유형 쿨다운 여부 확인
#   check()                 - 매 프레임 위반 항목 종합 판정 후 콜백 호출
#   run()                   - 영상 캡처 및 YOLO 추적 루프 실행
# ──────────────────────────────────────────────
class DecideViolation:

    RIDER_LABELS = {"person_with_kickboard", "2-person_with_kickboard"}
    HELMET_LABEL = "helmet_X"
    DOUBLE_LABEL = "2-person_with_kickboard"

    # 함수 이름 : __init__()
    # 기능      : DecideViolation 객체를 초기화한다.
    # 파라미터  : YOLO       model        -> 이미 로드된 YOLO 모델 인스턴스
    #             ZoneDrawer zone_drawer  -> Zone 정보를 가진 ZoneDrawer 인스턴스
    #             callable   on_violation -> 위반 감지 시 호출할 콜백 함수
    # 반환값    : 없음
    def __init__(self, model: YOLO, zone_drawer: "ZoneDrawer", on_violation: callable):
        self.model        = model
        self.zone_drawer  = zone_drawer
        self.on_violation = on_violation
        self._last_alert  = {}   # {(track_id, violation_type): 마지막 알림 시각}
        self._font        = self._load_font(20)   # 한글 폰트를 한 번만 로드한다.

    # 함수 이름 : _load_font()
    # 기능      : 시스템에서 사용 가능한 한글 폰트를 순서대로 탐색하여 로드한다.
    #             찾지 못하면 PIL 기본 폰트를 반환한다.
    # 파라미터  : int size -> 폰트 크기 (픽셀)
    # 반환값    : ImageFont 객체
    @staticmethod
    def _load_font(size: int):
        candidates = [
            "malgunbd.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    # 함수 이름 : _draw_violations()
    # 기능      : 한 프레임에서 발생한 모든 위반 텍스트를 PIL 변환 1회로 일괄 렌더링한다.
    #             기존에는 위반 항목마다 BGR↔RGB 변환을 반복했으나,
    #             이 함수는 변환을 한 번만 수행하여 성능을 개선한다.
    # 파라미터  : np.ndarray frame          -> 렌더링할 대상 프레임 (BGR 이미지)
    #             list       render_targets -> [(x1, y1, [v_type, ...]), ...] 형태의 렌더링 목록
    # 반환값    : 없음
    def _draw_violations(self, frame: np.ndarray, render_targets: list):
        if not render_targets:
            return

        # BGR → RGB 변환을 한 번만 수행한다.
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)

        for (x1, y1, violations) in render_targets:
            for idx, v_type in enumerate(violations):
                draw.text(
                    (x1, y1 - 60 - idx * 28),
                    v_type,
                    font  = self._font,
                    fill  = (255, 0, 0),   # RGB 빨간색
                )

        # RGB → BGR 변환을 한 번만 수행하여 원본 프레임에 반영한다.
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        np.copyto(frame, result)

    # 함수 이름 : in_zone()
    # 기능      : 바운딩 박스의 하단 중심(발 위치)이 특정 Zone 내부인지 확인한다.
    # 파라미터  : dict zone, int x1, int y1, int x2, int y2
    # 반환값    : bool -> True 이면 발 위치가 Zone 내부
    def in_zone(self, zone: dict, x1: int, y1: int, x2: int, y2: int) -> bool:
        poly   = np.array(zone["pts"], dtype=np.int32)
        cx, cy = int((x1 + x2) / 2), int(y2)
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

    # 함수 이름 : _is_helmet_missing()
    # 기능      : helmet_X 박스 중심이 탑승자 박스 내부에 있으면 헬멧 미착용으로 판정한다.
    # 파라미터  : list boxes     -> 전체 바운딩 박스 목록
    #             list labels    -> 각 박스에 대응하는 레이블 목록
    #             int  rider_idx -> 탑승자 박스의 인덱스
    # 반환값    : bool -> True 이면 헬멧 미착용
    def _is_helmet_missing(self, boxes: list, labels: list, rider_idx: int) -> bool:
        rx1, ry1, rx2, ry2 = boxes[rider_idx]
        for i, label in enumerate(labels):
            if label != self.HELMET_LABEL:
                continue
            hx1, hy1, hx2, hy2 = boxes[i]
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            if rx1 <= hcx <= rx2 and ry1 <= hcy <= ry2:
                return True
        return False

    # 함수 이름 : _is_sidewalk_riding()
    # 기능      : 탑승자 박스 하단 중심이 Zone 안에 있으면 인도주행으로 판정한다.
    # 파라미터  : int x1, y1, x2, y2 -> 탑승자 바운딩 박스 좌표
    # 반환값    : bool -> True 이면 인도주행
    def _is_sidewalk_riding(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        for zone in self.zone_drawer.zones:
            if self.in_zone(zone, x1, y1, x2, y2):
                return True
        return False

    # 함수 이름 : _is_double_riding()
    # 기능      : 2-person_with_kickboard 레이블이면 다인탑승으로 판정한다.
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
        os.makedirs(VIOLATION_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{violation_type}_{track_id}.jpg"
        filepath  = os.path.join(VIOLATION_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    # 함수 이름 : _should_alert()
    # 기능      : 동일 객체·위반 유형 조합에 대해 쿨다운이 지났는지 확인한다.
    #             time 을 파일 상단에서 import 하여 매 호출마다 import 하지 않는다. (최적화)
    # 파라미터  : int track_id       -> 객체 추적 ID
    #             str violation_type -> 위반 유형 문자열
    # 반환값    : bool -> True 이면 알림 전송 가능
    def _should_alert(self, track_id: int, violation_type: str) -> bool:
        key = (track_id, violation_type)
        if time.time() - self._last_alert.get(key, 0) >= COOLDOWN:
            self._last_alert[key] = time.time()
            return True
        return False

    # 함수 이름 : check()
    # 기능      : 한 프레임의 YOLO 감지 결과를 받아 위반 항목을 종합 판정한다.
    #             텍스트 렌더링은 _draw_violations() 로 일괄 처리하여 PIL 변환을 1회로 줄인다. (최적화)
    #             _save_frame() 은 쿨다운 통과 후에만 호출한다. (최적화)
    # 파라미터  : np.ndarray frame  -> 현재 처리 중인 영상 프레임
    #             list       boxes  -> 바운딩 박스 좌표 목록 [(x1,y1,x2,y2), ...]
    #             list       labels -> 각 박스에 대응하는 레이블 문자열 목록
    #             list       confs  -> 각 박스에 대응하는 신뢰도 목록
    #             list       ids    -> 각 박스에 대응하는 추적 ID 목록
    # 반환값    : 없음
    def check(self, frame: np.ndarray, boxes: list, labels: list,
              confs: list, ids: list):

        # 이번 프레임에서 렌더링할 위반 텍스트를 수집한다.
        render_targets = []   # [(x1, y1, [v_type, ...]), ...]

        for i, label in enumerate(labels):
            if label not in self.RIDER_LABELS:
                continue

            x1, y1, x2, y2 = boxes[i]
            tid  = ids[i]
            conf = confs[i]

            violations = []

            if self._is_helmet_missing(boxes, labels, i):
                violations.append("헬멧 미착용")

            if self._is_sidewalk_riding(x1, y1, x2, y2):
                violations.append("인도주행")

            if self._is_double_riding(label):
                violations.append("다인탑승")

            if violations:
                render_targets.append((x1, y1, violations))

            for v_type in violations:
                if self._should_alert(tid, v_type):
                    # 쿨다운 통과 후에만 프레임을 저장한다. (최적화)
                    # saved_path = self._save_frame(frame, v_type, tid)  # 추후 image_url 연동 시 활성화

                    self.on_violation(
                        violation_type = v_type,
                        track_id       = tid,
                        conf           = conf,
                        # image_path   = saved_path,
                    )
                    print(f"[VIOLATION] {v_type} | #{tid} | conf={conf:.2f}")

        # PIL 변환을 한 번만 수행하여 모든 위반 텍스트를 일괄 렌더링한다. (최적화)
        self._draw_violations(frame, render_targets)

    # 함수 이름 : run()
    # 기능      : 영상을 프레임 단위로 읽고 YOLO 로 추적하면서 check() 를 호출한다.
    #             enumerate 로 인덱스를 직접 사용하여 boxes.index() 탐색을 제거한다. (최적화)
    #             처리된 프레임을 JPEG 로 인코딩하여 latest_frame 에 저장한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def run(self):
        global latest_frame

        cap = cv2.VideoCapture(SOURCE)
        print("\n[감지 루프 시작]\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = self.model.track(
                frame, persist=True, conf=CONF, verbose=False,
                tracker="bytetrack.yaml", vid_stride=3
            )

            self.zone_drawer.draw_zones(frame)

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

                # enumerate 로 인덱스를 직접 사용하여 boxes.index() 탐색을 제거한다. (최적화)
                for idx, (box, label, conf, tid) in enumerate(zip(boxes, labels, confs, ids)):
                    ann.box_label(box, f"{label} #{tid} {conf:.2f}",
                                  color=yolo_colors(cls_ids[idx], bgr=True))

                self.check(frame, boxes, labels, confs, ids)

            # JPEG 인코딩 파라미터를 전역 상수로 재사용한다. (최적화)
            _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
            latest_frame = buf.tobytes()

        cap.release()


# ──────────────────────────────────────────────
# ConnectAPI 클래스
# FastAPI 라우터와 엔드포인트를 하나로 묶는다.
#
# 내장 함수 목록:
#   __init__()          - FastAPI 앱, CORS, 라우터를 초기화하고 엔드포인트를 등록한다
#   send_violation()    - 위반 정보를 별도 스레드로 백엔드에 전송한다 (논블로킹)
#   _post_violation()   - 백엔드 POST /api/violations 실제 전송 (스레드에서 호출)
#   video_stream()      - latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다
#   get_zones()         - 현재 설정된 Zone 목록을 반환한다
#   set_zones()         - React 캔버스에서 그린 Zone 목록을 받아 저장한다
#   delete_zones()      - 모든 Zone 을 초기화하고 파일을 갱신한다
#   get_alerts()        - 누적된 위반 알림 목록을 반환한다
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

        self.app.get("/video/stream")(self.video_stream)
        self.app.get("/zones")(self.get_zones)
        self.app.post("/zones")(self.set_zones)
        self.app.delete("/zones")(self.delete_zones)
        self.app.get("/alerts")(self.get_alerts)

    # 함수 이름 : send_violation()
    # 기능      : 위반 정보를 alert_history 에 추가하고
    #             백엔드 전송을 별도 스레드에서 실행하여 감지 루프를 블로킹하지 않는다.
    # 파라미터  : str   violation_type -> 위반 유형
    #             int   track_id       -> 객체 추적 ID
    #             float conf           -> YOLO 감지 신뢰도 (0.0 ~ 1.0)
    # 반환값    : 없음
    def send_violation(self, violation_type: str, track_id: int, conf: float):
        payload = {
            "type":       violation_type,
            "image_url":  "",   # TODO: 나중에 실제 image_url 로 교체
            "camera":     CAMERA_ID,
            "confidence": int(conf * 100),
        }

        alert = {
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "type":       violation_type,
            "track_id":   track_id,
            "confidence": payload["confidence"],
            "camera":     CAMERA_ID,
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
                await asyncio.sleep(0.03)
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

    model = YOLO(MODEL_PATH).to("cuda")
    print(model.device)
    print("YOLO model loaded successfully!")

    drawer = ZoneDrawer()
    if os.path.exists(ZONE_FILE):
        drawer.load(ZONE_FILE)

    api = ConnectAPI(drawer=drawer, alert_history=alert_history)

    detector = DecideViolation(
        model        = model,
        zone_drawer  = drawer,
        on_violation = api.send_violation,
    )

    Thread(target=detector.run, daemon=True).start()

    uvicorn.run(api.app, host="0.0.0.0", port=8000)