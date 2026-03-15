import ultralytics
#from ultralytics import YOLO
import cv2


#model = YOLO('best_v3.pt')  # nano 모델 (가장 빠름)
#print("YOLO model loaded successfully!")

#results = model.track(source="1.mp4",show = True, conf=0.5)


import cv2
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as yolo_colors

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SOURCE     = "1.mp4"
MODEL_PATH = "best_v3.pt"
CONF       = 0.5
COOLDOWN   = 3.0
SAVE_PATH  = "output.mp4"
ZONE_FILE  = "zones.json"

COLORS = [
    (0,   0,   255),
    (0,  165,  255),
    (0,  255,    0),
    (255, 100,   0),
    (255,   0,  200),
]

WIN = "Kickboard Zone Monitor"


# ──────────────────────────────────────────────
# 전역 Zone 상태
# ──────────────────────────────────────────────
zones       = []       # 완성된 Zone 목록
_pts        = []       # 현재 그리는 꼭짓점
_mouse      = (0, 0)
_cidx       = 0
_zone_num   = 1
_draw_mode  = False    # True = Zone 그리기 모드 (일시정지 상태)
_base_frame = None     # 일시정지된 프레임 (그리기 배경)


def _color():
    return COLORS[_cidx % len(COLORS)]


def _finish_zone():
    global _pts, _cidx, _zone_num
    if len(_pts) < 3:
        print("[경고] 최소 3개 꼭짓점 필요")
        return
    name = f"Zone-{_zone_num}"
    zones.append({"name": name, "pts": list(_pts), "color": _color()})
    print(f"[✓] {name} 완성 ({len(_pts)}개 꼭짓점)")
    _pts = []
    _cidx += 1
    _zone_num += 1


# ──────────────────────────────────────────────
# 마우스 콜백 (그리기 모드일 때만 동작)
# ──────────────────────────────────────────────
def on_mouse(event, x, y, flags, _):
    global _pts, _mouse
    if not _draw_mode:
        return
    _mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _pts.append((x, y))
        print(f"  꼭짓점 ({x}, {y})  총 {len(_pts)}개")
    elif event == cv2.EVENT_RBUTTONDOWN:
        _finish_zone()


# ──────────────────────────────────────────────
# 렌더링 헬퍼
# ──────────────────────────────────────────────
def draw_zones(frame: np.ndarray):
    """완성된 Zone을 반투명으로 그림."""
    for z in zones:
        poly = np.array(z["pts"], dtype=np.int32)
        ov   = frame.copy()
        cv2.fillPoly(ov, [poly], z["color"])
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [poly], True, z["color"], 2)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, z["name"], (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)


def draw_current(frame: np.ndarray):
    """현재 그리는 꼭짓점 + 미리보기 선."""
    color = _color()
    if _pts:
        poly = np.array(_pts, dtype=np.int32)
        cv2.polylines(frame, [poly], False, color, 2)
        for p in _pts:
            cv2.circle(frame, p, 5, color, -1)
        cv2.line(frame, _pts[-1], _mouse, color, 1)
        if len(_pts) >= 3:
            _dashed(frame, _mouse, _pts[0], color)
    mx, my = _mouse
    cv2.line(frame, (mx - 14, my), (mx + 14, my), color, 1)
    cv2.line(frame, (mx, my - 14), (mx, my + 14), color, 1)


def draw_hud(frame: np.ndarray):
    h = frame.shape[0]
    if _draw_mode:
        lines = [
            f"[일시정지 - 구역 그리기]  완성: {len(zones)}개  현재 꼭짓점: {len(_pts)}개",
            "좌클릭=점추가  우클릭=Zone완성  n=새구역  z=되돌리기  SPACE=재생재개  q=종료",
        ]
    else:
        lines = [
            f"[재생 중]  설정된 Zone: {len(zones)}개",
            "SPACE=일시정지(구역설정)  s=Zone저장  q=종료",
        ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, h - 36 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def _dashed(img, p1, p2, color, gap=8):
    x1, y1 = p1; x2, y2 = p2
    dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    if dist == 0: return
    n = max(int(dist / gap), 1)
    for i in range(0, n, 2):
        t1, t2 = i/n, min((i+1)/n, 1.0)
        cv2.line(img,
                 (int(x1+(x2-x1)*t1), int(y1+(y2-y1)*t1)),
                 (int(x1+(x2-x1)*t2), int(y1+(y2-y1)*t2)),
                 color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Zone 저장 / 불러오기
# ──────────────────────────────────────────────
def save_zones(path: str = ZONE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "zones": [{"name": z["name"],
                               "pts":  z["pts"],
                               "color": list(z["color"])} for z in zones]}, f, indent=2)
    print(f"[저장] {len(zones)}개 Zone → {path}")


def load_zones(path: str = ZONE_FILE):
    global zones
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    zones = [{"name": z["name"],
              "pts":  [tuple(p) for p in z["pts"]],
              "color": tuple(z["color"])} for z in data["zones"]]
    print(f"[불러오기] {len(zones)}개 Zone ← {path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    global _draw_mode, _base_frame, _pts, _cidx, _zone_num

    # 모델 로드
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully!")

    # 저장된 Zone 불러오기 (있으면)
    import os, time
    if os.path.exists(ZONE_FILE):
        ans = input(f"저장된 Zone({ZONE_FILE})을 불러올까요? [y/n]: ").strip().lower()
        if ans == 'y':
            load_zones()

    # 알림 쿨다운
    last_alert: dict = {}

    def should_alert(zone_name, tid):
        key = (zone_name, tid)
        if time.time() - last_alert.get(key, 0) >= COOLDOWN:
            last_alert[key] = time.time()
            return True
        return False

    # 캡처 & 저장 설정
    cap    = cv2.VideoCapture(SOURCE)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_res  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = (cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h_res))
              if SAVE_PATH else None)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    print("\n[실행 중]  SPACE=일시정지/구역설정  s=저장  q=종료\n")

    while True:

        # ── 그리기 모드 (일시정지) ────────────────────
        if _draw_mode:
            frame = _base_frame.copy()
            draw_zones(frame)
            draw_current(frame)
            draw_hud(frame)
            cv2.imshow(WIN, frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):                    # 재생 재개
                _draw_mode = False
                print("[재생 재개]")
            elif key == ord('n'):                  # 새 Zone
                if len(_pts) >= 3: _finish_zone()
                else: _pts = []; _cidx += 1
            elif key == ord('z'):                  # 되돌리기
                if _pts: print(f"  되돌리기: {_pts.pop()}")
            elif key == ord('c'):                  # 현재 초기화
                _pts = []
            elif key == ord('r'):                  # 전체 초기화
                _pts = []; zones.clear()
                _cidx = 0; _zone_num = 1
                print("[전체 초기화]")
            elif key == ord('s'):                  # 저장
                if len(_pts) >= 3: _finish_zone()
                save_zones()
            elif key in (ord('q'), 27):
                break
            continue

        # ── 재생 모드 ─────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 추적
        results = model.track(frame, persist=True, conf=CONF, verbose=False)

        # Zone 오버레이
        draw_zones(frame)

        # 객체 처리
        if results[0].boxes is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            ids     = (results[0].boxes.id.cpu().numpy().astype(int)
                       if results[0].boxes.id is not None
                       else range(len(boxes)))
            ann = Annotator(frame, line_width=2)

            for box, cid, conf, tid in zip(boxes, cls_ids, confs, ids):
                x1, y1, x2, y2 = map(int, box)
                ann.box_label((x1, y1, x2, y2),
                              f"{model.names[cid]} #{tid} {conf:.2f}",
                              color=yolo_colors(cid, bgr=True))

                for z in zones:
                    poly = np.array(z["pts"], dtype=np.int32)
                    cx, cy = int((x1+x2)/2), int(y2)
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"! {model.names[cid]} IN {z['name']}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 0, 255), 2)
                        if should_alert(z["name"], int(tid)):
                            print(f"[ALERT] {datetime.now().strftime('%H:%M:%S')} | "
                                  f"{z['name']} | #{tid} {model.names[cid]} ({conf:.2f})")

        draw_hud(frame)

        if writer:
            writer.write(frame)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):                        # 일시정지 → 그리기 모드
            _draw_mode  = True
            _base_frame = frame.copy()
            print("[일시정지]  마우스로 Zone을 그리세요  SPACE=재생재개")
        elif key == ord('s'):
            save_zones()
        elif key in (ord('q'), 27):
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("[완료]")


if __name__ == "__main__":
    main()