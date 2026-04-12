import cv2
import json
import asyncio
import time
import numpy as np
import httpx
import os
from abc import ABC, abstractmethod
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
ENCODE_PARAMS   = [cv2.IMWRITE_JPEG_QUALITY, 65]   # JPEG 인코딩 파라미터 — 재사용

# ──────────────────────────────────────────────
# 전역 공유 상태
# ──────────────────────────────────────────────
latest_frame: bytes = b""   # MJPEG 스트림용 최신 프레임
alert_history: list = []    # 누적 알림 목록


# ──────────────────────────────────────────────
# ZoneDrawer 클래스 — Zone 데이터 관리 전담
# (렌더링 책임은 ZoneRenderer 로 분리)
#
# 내장 함수 목록:
#   __init__()    - Zone 상태 변수 초기화
#   _color()      - 현재 색상 인덱스에 해당하는 BGR 색상 튜플 반환
#   finish_zone() - 현재 꼭짓점으로 Zone 완성하여 zones 목록에 추가
#   save()        - 현재 zones 목록을 JSON 파일로 저장
#   load()        - JSON 파일에서 Zone 목록을 불러와 zones 에 저장
#   set_zones()   - API 에서 받은 좌표 목록으로 zones 를 교체한다
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
    # 기능      : API 에서 받은 Zone 좌표 목록으로 zones 를 교체하고 파일에 저장한다.
    #             color 가 없으면 팔레트에서 자동 배정한다.
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
# ZoneRenderer 클래스 — Zone 렌더링 전담 (SRP 분리)
#
# 내장 함수 목록:
#   __init__()     - ZoneDrawer 참조 초기화
#   draw_zones()   - 완성된 모든 Zone을 반투명 채우기와 외곽선으로 프레임에 표시
#   _draw_dashed() - 두 점 사이를 일정 간격의 점선으로 표시
# ──────────────────────────────────────────────
class ZoneRenderer:

    # 함수 이름 : __init__()
    # 기능      : ZoneRenderer 를 초기화한다.
    # 파라미터  : ZoneDrawer zone_drawer -> Zone 데이터를 가진 ZoneDrawer 인스턴스
    # 반환값    : 없음
    def __init__(self, zone_drawer: "ZoneDrawer"):
        self.zone_drawer = zone_drawer

    # 함수 이름 : draw_zones()
    # 기능      : zones 목록에 있는 모든 완성된 Zone을
    #             반투명 채우기와 외곽선으로 프레임에 그린다.
    # 파라미터  : np.ndarray frame -> 그림을 그릴 대상 프레임 (BGR 이미지)
    # 반환값    : 없음
    def draw_zones(self, frame: np.ndarray):
        for z in self.zone_drawer.zones:
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


# ──────────────────────────────────────────────
# ViolationStrategy 추상 클래스 — 전략 패턴 (OCP 적용)
# 새 위반 유형 추가 시 이 클래스를 상속하여 구현한다.
#
# 내장 함수 목록:
#   check() - 위반 여부를 판정하여 위반 유형 문자열을 반환한다 (추상 메서드)
# ──────────────────────────────────────────────
class ViolationStrategy(ABC):

    # 함수 이름 : check()
    # 기능      : 한 프레임의 감지 결과에서 해당 전략의 위반 여부를 판정한다.
    # 파라미터  : list boxes    -> 전체 바운딩 박스 목록 [(x1,y1,x2,y2), ...]
    #             list labels   -> 각 박스에 대응하는 레이블 문자열 목록
    #             int  idx      -> 현재 탑승자 박스의 인덱스
    # 반환값    : str | None -> 위반 유형 문자열 또는 None (위반 없음)
    @abstractmethod
    def check(self, boxes: list, labels: list, idx: int) -> str | None:
        pass


# ──────────────────────────────────────────────
# HelmetViolation 클래스 — 헬멧 미착용 판정 전략
# ──────────────────────────────────────────────
class HelmetViolation(ViolationStrategy):

    HELMET_LABEL = "helmet_X"

    # 함수 이름 : check()
    # 기능      : helmet_X 박스 중심이 탑승자 박스 내부에 있으면 헬멧 미착용으로 판정한다.
    # 파라미터  : list boxes -> 전체 바운딩 박스 목록
    #             list labels -> 각 박스에 대응하는 레이블 목록
    #             int  idx    -> 탑승자 박스의 인덱스
    # 반환값    : str | None -> "헬멧 미착용" 또는 None
    def check(self, boxes: list, labels: list, idx: int) -> str | None:
        rx1, ry1, rx2, ry2 = boxes[idx]
        for i, label in enumerate(labels):
            if label != self.HELMET_LABEL:
                continue
            hx1, hy1, hx2, hy2 = boxes[i]
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            if rx1 <= hcx <= rx2 and ry1 <= hcy <= ry2:
                return "헬멧 미착용"
        return None


# ──────────────────────────────────────────────
# SidewalkViolation 클래스 — 인도주행 판정 전략
# ──────────────────────────────────────────────
class SidewalkViolation(ViolationStrategy):

    # 함수 이름 : __init__()
    # 기능      : SidewalkViolation 을 초기화한다.
    # 파라미터  : ZoneDrawer zone_drawer -> Zone 정보를 가진 ZoneDrawer 인스턴스
    # 반환값    : 없음
    def __init__(self, zone_drawer: "ZoneDrawer"):
        self.zone_drawer = zone_drawer

    # 함수 이름 : check()
    # 기능      : 탑승자 박스 하단 중심이 Zone 안에 있으면 인도주행으로 판정한다.
    # 파라미터  : list boxes -> 전체 바운딩 박스 목록
    #             list labels -> 각 박스에 대응하는 레이블 목록 (미사용)
    #             int  idx    -> 탑승자 박스의 인덱스
    # 반환값    : str | None -> "인도주행" 또는 None
    def check(self, boxes: list, labels: list, idx: int) -> str | None:
        x1, y1, x2, y2 = boxes[idx]
        cx, cy = int((x1 + x2) / 2), int(y2)
        for zone in self.zone_drawer.zones:
            poly = np.array(zone["pts"], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                return "인도주행"
        return None


# ──────────────────────────────────────────────
# DoubleRidingViolation 클래스 — 다인탑승 판정 전략
# ──────────────────────────────────────────────
class DoubleRidingViolation(ViolationStrategy):

    DOUBLE_LABEL = "2-person_with_kickboard"

    # 함수 이름 : check()
    # 기능      : 2-person_with_kickboard 레이블이면 다인탑승으로 판정한다.
    # 파라미터  : list boxes  -> 전체 바운딩 박스 목록 (미사용)
    #             list labels -> 각 박스에 대응하는 레이블 목록
    #             int  idx    -> 탑승자 박스의 인덱스
    # 반환값    : str | None -> "다인탑승" 또는 None
    def check(self, boxes: list, labels: list, idx: int) -> str | None:
        if labels[idx] == self.DOUBLE_LABEL:
            return "다인탑승"
        return None


# ──────────────────────────────────────────────
# DecideViolation 클래스 — 위반 판정 전담
# 전략 목록을 순서대로 실행하고 쿨다운·렌더링·콜백을 처리한다.
# 영상 루프는 DetectionLoop 으로 분리되어 이 클래스에 없다.
#
# 내장 함수 목록:
#   __init__()          - 전략 목록, 쿨다운, 폰트, 콜백 초기화
#   _load_font()        - 시스템 한글 폰트 탐색 및 로드
#   _should_alert()     - 동일 객체·위반 유형 쿨다운 여부 확인
#   _save_frame()       - 위반 프레임을 이미지 파일로 저장
#   _draw_violations()  - 위반 텍스트를 PIL 변환 1회로 일괄 렌더링
#   check()             - 매 프레임 전략 목록을 실행하여 위반 종합 판정
# ──────────────────────────────────────────────
class DecideViolation:

    RIDER_LABELS = {"person_with_kickboard", "2-person_with_kickboard"}

    # 함수 이름 : __init__()
    # 기능      : DecideViolation 객체를 초기화한다.
    # 파라미터  : list     strategies   -> ViolationStrategy 인스턴스 목록
    #             callable on_violation -> 위반 감지 시 호출할 콜백 함수
    # 반환값    : 없음
    def __init__(self, strategies: list, on_violation: callable):
        self.strategies   = strategies
        self.on_violation = on_violation
        self._last_alert  = {}             # {(track_id, violation_type): 마지막 알림 시각}
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

    # 함수 이름 : _should_alert()
    # 기능      : 동일 객체·위반 유형 조합에 대해 쿨다운이 지났는지 확인한다.
    # 파라미터  : int track_id       -> 객체 추적 ID
    #             str violation_type -> 위반 유형 문자열
    # 반환값    : bool -> True 이면 알림 전송 가능
    def _should_alert(self, track_id: int, violation_type: str) -> bool:
        key = (track_id, violation_type)
        if time.time() - self._last_alert.get(key, 0) >= COOLDOWN:
            self._last_alert[key] = time.time()
            return True
        return False

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

    # 함수 이름 : _draw_violations()
    # 기능      : 한 프레임에서 발생한 모든 위반 텍스트를 PIL 변환 1회로 일괄 렌더링한다.
    # 파라미터  : np.ndarray frame          -> 렌더링할 대상 프레임 (BGR 이미지)
    #             list       render_targets -> [(x1, y1, [v_type, ...]), ...] 렌더링 목록
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
                    font = self._font,
                    fill = (255, 0, 0),   # RGB 빨간색
                )

        # RGB → BGR 변환을 한 번만 수행하여 원본 프레임에 반영한다.
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        np.copyto(frame, result)

    # 함수 이름 : check()
    # 기능      : 한 프레임의 YOLO 감지 결과에서 전략 목록을 순서대로 실행하여
    #             위반 항목을 종합 판정하고 콜백을 호출한다.
    #             새 위반 유형 추가 시 이 메서드를 수정하지 않아도 된다. (OCP)
    # 파라미터  : np.ndarray frame  -> 현재 처리 중인 영상 프레임
    #             list       boxes  -> 바운딩 박스 좌표 목록 [(x1,y1,x2,y2), ...]
    #             list       labels -> 각 박스에 대응하는 레이블 문자열 목록
    #             list       confs  -> 각 박스에 대응하는 신뢰도 목록
    #             list       ids    -> 각 박스에 대응하는 추적 ID 목록
    # 반환값    : 없음
    def check(self, frame: np.ndarray, boxes: list, labels: list,
              confs: list, ids: list):

        render_targets = []   # [(x1, y1, [v_type, ...]), ...]

        for i, label in enumerate(labels):
            if label not in self.RIDER_LABELS:
                continue

            x1, y1, x2, y2 = boxes[i]
            tid  = ids[i]
            conf = confs[i]

            # 전략 목록을 순서대로 실행하여 위반 항목을 수집한다.
            violations = [
                result
                for strategy in self.strategies
                if (result := strategy.check(boxes, labels, i)) is not None
            ]

            if violations:
                render_targets.append((x1, y1, violations))

            for v_type in violations:
                if self._should_alert(tid, v_type):
                    # saved_path = self._save_frame(frame, v_type, tid)  # 추후 image_url 연동 시 활성화
                    self.on_violation(
                        violation_type = v_type,
                        track_id       = tid,
                        conf           = conf,
                        # image_path   = saved_path,
                    )
                    print(f"[VIOLATION] {v_type} | #{tid} | conf={conf:.2f}")

        # PIL 변환을 한 번만 수행하여 모든 위반 텍스트를 일괄 렌더링한다.
        self._draw_violations(frame, render_targets)


# ──────────────────────────────────────────────
# DetectionLoop 클래스 — 영상 캡처 및 추론 루프 전담 (SRP 분리)
# DecideViolation 에서 루프 책임을 분리한다.
#
# 내장 함수 목록:
#   __init__() - 모델, 렌더러, 판정기 초기화
#   run()      - 영상 캡처, YOLO 추론, 프레임 인코딩 루프 실행
# ──────────────────────────────────────────────
class DetectionLoop:

    # 함수 이름 : __init__()
    # 기능      : DetectionLoop 객체를 초기화한다.
    # 파라미터  : YOLO            model    -> 이미 로드된 YOLO 모델 인스턴스
    #             ZoneRenderer    renderer -> Zone 렌더링 담당 인스턴스
    #             DecideViolation decider  -> 위반 판정 담당 인스턴스
    # 반환값    : 없음
    def __init__(self, model: YOLO, renderer: "ZoneRenderer", decider: "DecideViolation"):
        self.model    = model
        self.renderer = renderer
        self.decider  = decider

    # 함수 이름 : run()
    # 기능      : 영상을 프레임 단위로 읽고 YOLO 로 추적하면서 decider.check() 를 호출한다.
    #             처리된 프레임을 JPEG 로 인코딩하여 latest_frame 전역 변수에 저장한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def run(self):
        global latest_frame

        cap = cv2.VideoCapture(SOURCE)
        print("\n[감지 루프 시작]\n")

        while True:
            ret, frame = cap.read()
            if not ret:                         # 영상 끝에 도달하면 처음부터 재생한다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # YOLO 로 객체를 추적한다.
            results = self.model.track(
                frame, persist=True, conf=CONF, verbose=False,
                tracker="bytetrack.yaml", vid_stride=3
            )

            self.renderer.draw_zones(frame)     # Zone 오버레이 렌더링

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
                for idx, (box, label, conf, tid) in enumerate(zip(boxes, labels, confs, ids)):
                    ann.box_label(box, f"{label} #{tid} {conf:.2f}",
                                  color=yolo_colors(cls_ids[idx], bgr=True))

                # 위반 판정을 수행한다.
                self.decider.check(frame, boxes, labels, confs, ids)

            # 프레임을 JPEG 로 인코딩하여 스트리밍용 전역 변수에 저장한다.
            _, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
            latest_frame = buf.tobytes()

        cap.release()


# ──────────────────────────────────────────────
# ConnectAPI 클래스 — API 통신 전담
#
# 내장 함수 목록:
#   __init__()          - FastAPI 앱, CORS, 엔드포인트 등록
#   send_violation()    - 위반 정보를 별도 스레드로 백엔드에 전송 (논블로킹)
#   _post_violation()   - 백엔드 POST /api/violations 실제 전송 (스레드에서 호출)
#   video_stream()      - latest_frame 을 MJPEG 형식으로 실시간 스트리밍
#   get_zones()         - 현재 설정된 Zone 목록 반환
#   set_zones()         - React 캔버스에서 그린 Zone 목록을 받아 저장
#   delete_zones()      - 모든 Zone 초기화 후 파일 갱신
#   get_alerts()        - 누적된 위반 알림 목록 반환
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
    # 기능      : React 캔버스에서 그린 Zone 목록을 받아 ZoneDrawer 에 저장한다.
    #             Zone 변환 로직은 ZoneDrawer.set_zones() 에서 처리한다. (SRP)
    # 파라미터  : body = {"zones": [{"name": str, "pts": [[x,y],...], "color": [B,G,R]}, ...]}
    # 반환값    : {"saved": Zone 개수}
    async def set_zones(self, body: dict):
        self.drawer.set_zones(body.get("zones", []))   # 변환 로직은 ZoneDrawer 에 위임
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
    model = YOLO(MODEL_PATH).to("cuda")
    print(model.device)
    print("YOLO model loaded successfully!")

    # Zone 데이터 관리 객체를 생성하고 저장된 Zone 을 불러온다.
    drawer = ZoneDrawer()
    if os.path.exists(ZONE_FILE):
        drawer.load(ZONE_FILE)

    # Zone 렌더링 객체를 생성한다.
    renderer = ZoneRenderer(drawer)

    # API 통신 객체를 생성한다.
    api = ConnectAPI(drawer=drawer, alert_history=alert_history)

    # 위반 판정 전략 목록을 구성한다.
    # 새 위반 유형 추가 시 이 목록에만 추가하면 된다. (OCP)
    strategies = [
        HelmetViolation(),
        SidewalkViolation(drawer),
        DoubleRidingViolation(),
    ]

    # 위반 판정 객체를 생성한다.
    decider = DecideViolation(
        strategies   = strategies,
        on_violation = api.send_violation,
    )

    # 감지 루프 객체를 생성하고 별도 스레드에서 실행한다.
    loop = DetectionLoop(model=model, renderer=renderer, decider=decider)
    Thread(target=loop.run, daemon=True).start()

    # FastAPI 서버를 실행한다.
    uvicorn.run(api.app, host="0.0.0.0", port=8000)