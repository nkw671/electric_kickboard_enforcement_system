import cv2
import json
import asyncio
import numpy as np
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
SOURCE     = "src/1.mp4"              # 입력 영상 경로 (웹캠은 0)
MODEL_PATH = "src/best_v3.pt"     # YOLO 모델 가중치 경로
CONF       = 0.5                  # 객체 감지 최소 신뢰도
COOLDOWN   = 3.0                  # 동일 객체 재알림 최소 간격 (초)
ZONE_FILE  = "zones.json"         # Zone 좌표 저장/불러오기 파일 경로

# ──────────────────────────────────────────────
# 전역 공유 상태
# ──────────────────────────────────────────────
latest_frame: bytes = b""         # MJPEG 스트림용 최신 프레임
alert_history: list  = []         # 누적 알림 목록 (GET /alerts 용)
drawer = None # run_detection() 에서 초기화 후 ZoneAPI 와 공유


# ──────────────────────────────────────────────
# ZoneDrawer 클래스 (서버용 — GUI 제거, 로직만 유지)
#
# 재사용  : draw_zones / in_zone / save / load / finish_zone / _color / _draw_dashed
# 제거    : on_mouse / enter_draw_mode / exit_draw_mode / handle_key
#           render / _draw_current / _draw_hud / _put_korean_text / _load_font
#
# 내장 함수 목록:
#   __init__()     - Zone 상태 변수 초기화
#   _color()       - 현재 색상 인덱스에 해당하는 BGR 색상 튜플 반환
#   finish_zone()  - 현재 꼭짓점으로 Zone 완성하여 zones 목록에 추가
#   draw_zones()   - 완성된 모든 Zone을 반투명 채우기와 외곽선으로 프레임에 표시
#   _draw_dashed() - 두 점 사이를 일정 간격의 점선으로 표시
#   in_zone()      - 바운딩 박스 하단 중심이 특정 Zone 내부인지 확인
#   save()         - 현재 zones 목록을 JSON 파일로 저장
#   load()         - JSON 파일에서 Zone 목록을 불러와 zones에 저장
#   set_zones()    - API 에서 받은 좌표 목록으로 zones 를 교체한다
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

    # 함수 이름 : in_zone()
    # 기능      : 바운딩 박스의 하단 중심(발 위치)이 특정 Zone 내부인지 확인한다.
    # 파라미터  : dict zone, int x1, int y1, int x2, int y2
    # 반환값    : bool -> True 이면 발 위치가 Zone 내부
    def in_zone(self, zone: dict, x1, y1, x2, y2) -> bool:
        poly   = np.array(zone["pts"], dtype=np.int32)
        cx, cy = int((x1 + x2) / 2), int(y2)
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

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
# 감지 루프 (기존 main() 핵심 로직 재사용, GUI 코드만 제거)
# ──────────────────────────────────────────────

# 함수 이름 : run_detection()
# 기능      : YOLO 로 객체를 추적하고 Zone 침범을 감지한다.
#             매 프레임을 JPEG 로 인코딩해 latest_frame 에 저장하여 스트리밍에 제공한다.
#             cv2.imshow 등 GUI 코드는 제거하고 나머지 감지 로직은 기존과 동일하다.
# 파라미터  : 없음
# 반환값    : 없음
def run_detection():
    global latest_frame, drawer
    import os, time
    # YOLO 모델을 로드한다.
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully!")

    # ZoneDrawer 객체를 생성하고 저장된 Zone 을 불러온다.
    drawer = ZoneDrawer()
    if os.path.exists(ZONE_FILE):
        drawer.load(ZONE_FILE)

    # app.state 에 drawer 를 등록하여 API 에서 접근할 수 있게 한다.


    # 알림 쿨다운 상태를 관리하는 딕셔너리를 초기화한다.
    last_alert: dict = {}

    # 함수 이름 : should_alert()
    # 기능      : 동일 Zone + 동일 객체에 대해 쿨다운이 지났는지 확인한다.
    # 파라미터  : str zone_name, int tid
    # 반환값    : bool
    def should_alert(zone_name, tid):
        key = (zone_name, tid)
        if time.time() - last_alert.get(key, 0) >= COOLDOWN:
            last_alert[key] = time.time()
            return True
        return False

    # 영상 캡처 객체를 초기화한다.
    cap = cv2.VideoCapture(SOURCE)

    print("\n[감지 루프 시작]\n")

    while True:
        ret, frame = cap.read()
        if not ret:                             # 영상 끝에 도달하면 처음부터 다시 재생한다.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # YOLO 로 객체를 추적한다.
        results = model.track(frame, persist=True, conf=CONF, verbose=False, tracker = "bytetrack.yaml", vid_stride=2)

        drawer.draw_zones(frame)                # Zone 오버레이 렌더링

        # 감지된 객체가 있을 때만 처리한다.
        if results[0].boxes is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            ids     = (
                results[0].boxes.id.cpu().numpy().astype(int)
                if results[0].boxes.id is not None
                else range(len(boxes))
            )
            ann = Annotator(frame, line_width=2)

            for box, cid, conf, tid in zip(boxes, cls_ids, confs, ids):
                x1, y1, x2, y2 = map(int, box)

                ann.box_label(
                    (x1, y1, x2, y2),
                    f"{model.names[cid]} #{tid} {conf:.2f}",
                    color=yolo_colors(cid, bgr=True),
                )

                for z in drawer.zones:
                    if drawer.in_zone(z, x1, y1, x2, y2):     # 기존 in_zone() 재사용
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame,
                            f"! {model.names[cid]}  in walkway!!",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                        )

                        if should_alert(z["name"], int(tid)):
                            alert = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "zone":      z["name"],
                                "track_id":  int(tid),
                                "class":     model.names[cid],
                                "conf":      round(float(conf), 2),
                            }
                            alert_history.append(alert)
                            print(f"[ALERT] {alert}")

        # 프레임을 JPEG 로 인코딩하여 스트리밍용 전역 변수에 저장한다.
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = buf.tobytes()

    cap.release()


# ──────────────────────────────────────────────
# ZoneAPI 클래스
# FastAPI 라우터와 엔드포인트를 하나로 묶는다.
#
# 내장 함수 목록:
#   __init__()      - FastAPI 앱, CORS, 라우터를 초기화하고 엔드포인트를 등록한다
#   video_stream()  - latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다
#   get_zones()     - 현재 설정된 Zone 목록을 반환한다
#   set_zones()     - React 캔버스에서 그린 Zone 목록을 받아 저장한다
#   delete_zones()  - 모든 Zone 을 초기화하고 파일을 갱신한다
#   get_alerts()    - 누적된 침범 알림 목록을 반환한다
# ──────────────────────────────────────────────
app = FastAPI()
class ZoneAPI:

    # 함수 이름 : __init__()
    # 기능      : FastAPI 앱을 생성하고 CORS 설정 및 엔드포인트를 등록한다.
    # 파라미터  : ZoneDrawer drawer      -> 감지 루프와 공유하는 ZoneDrawer 인스턴스
    #             list       alert_history -> 감지 루프와 공유하는 알림 목록
    # 반환값    : 없음
    def __init__(self, drawer: ZoneDrawer, alert_history: list):
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
        self.drawer.save(ZONE_FILE)             # ZoneDrawer.save() 재사용
        return {"saved": len(self.drawer.zones)}

    # 함수 이름 : delete_zones()
    # 기능      : 모든 Zone 을 초기화하고 파일을 갱신한다.
    # 파라미터  : 없음
    # 반환값    : {"cleared": true}
    async def delete_zones(self):
        self.drawer.zones = []
        self.drawer.save(ZONE_FILE)             # ZoneDrawer.save() 재사용
        return {"cleared": True}

    # 함수 이름 : get_alerts()
    # 기능      : 누적된 침범 알림 목록을 반환한다.
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

    # run_detection 이 알아서 drawer 를 생성하고 전역에 올린다.
    t = Thread(target=run_detection, daemon=True)
    t.start()
    t.join(timeout=3)  # drawer 초기화 대기

    api = ZoneAPI(drawer=drawer, alert_history=alert_history)
    uvicorn.run(api.app, host="0.0.0.0", port=8000)