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
SOURCE     = "1.mp4"       # 입력 영상 경로 (웹캠은 0)
MODEL_PATH = "best_v3.pt"  # 학습된 YOLO 모델 가중치 경로
CONF       = 0.5           # 객체 감지 최소 신뢰도
COOLDOWN   = 3.0           # 동일 객체 재알림 최소 간격 (초)
SAVE_PATH  = "none"  # 결과 영상 저장 경로
ZONE_FILE  = "zones.json"  # Zone 좌표 저장/불러오기 파일 경로

COLORS = [                 # Zone 색상 팔레트 (BGR 순서)
    (0,   0,   255),
    (0,  165,  255),
    (0,  255,    0),
    (255, 100,   0),
    (255,   0,  200),
]

WIN = "Kickboard Zone Monitor"  # OpenCV 창 이름


# ──────────────────────────────────────────────
# 전역 Zone 상태 변수
# ──────────────────────────────────────────────
zones       = []     # 완성된 Zone 딕셔너리 목록
_pts        = []     # 현재 그리는 중인 꼭짓점 좌표 목록
_mouse      = (0, 0) # 현재 마우스 커서 위치
_cidx       = 0      # 현재 선택된 색상 인덱스
_zone_num   = 1      # 다음 Zone 이름에 붙을 번호
_draw_mode  = False  # True 이면 일시정지 후 Zone 그리기 모드
_base_frame = None   # 일시정지 시 캡처한 배경 프레임


# 함수 이름 : _color()
# 기능      : 현재 색상 인덱스에 해당하는 BGR 색상 튜플을 반환한다.
# 파라미터  : 없음
# 반환값    : tuple -> COLORS 팔레트에서 선택된 (B, G, R) 색상 튜플
def _color():
    return COLORS[_cidx % len(COLORS)]


# 함수 이름 : _finish_zone()
# 기능      : 현재까지 찍은 꼭짓점으로 Zone을 완성하여 zones 목록에 추가한다.
#             꼭짓점이 3개 미만이면 경고 메시지를 출력하고 종료한다.
# 파라미터  : 없음
# 반환값    : 없음
def _finish_zone():
    global _pts, _cidx, _zone_num

    if len(_pts) < 3:                   # 다각형 성립하는지 최소 조건 검사
        print("[경고] 최소 3개 꼭짓점 필요")
        return

    # Zone 이름을 생성하고 zones 목록에 추가한다.
    name = f"Zone-{_zone_num}"
    zones.append({"name": name, "pts": list(_pts), "color": _color()})
    print(f"[✓] {name} 완성 ({len(_pts)}개 꼭짓점)")

    _pts = []        # 꼭짓점 목록 초기화
    _cidx += 1       # 다음 Zone 은 다른 색상 사용
    _zone_num += 1   # Zone 번호 증가


# 함수 이름 : on_mouse()
# 기능      : OpenCV 마우스 이벤트를 처리한다.
#             그리기 모드일 때만 동작하며,
#             좌클릭이면 꼭짓점을 추가하고 우클릭이면 Zone 을 완성한다.
# 파라미터  : int    event  -> OpenCV 마우스 이벤트 상수
#             int    x      -> 마우스 클릭 x 좌표 (픽셀)
#             int    y      -> 마우스 클릭 y 좌표 (픽셀)
#             int    flags  -> 추가 이벤트 플래그 (미사용)
#             object _      -> 사용자 데이터 (미사용)
# 반환값    : 없음
def on_mouse(event, x, y, flags, _):
    global _pts, _mouse

    if not _draw_mode:   # 재생 모드에서는 마우스 입력 무시
        return

    _mouse = (x, y)      # 현재 마우스 위치 갱신

    if event == cv2.EVENT_LBUTTONDOWN:       # 좌클릭 : 꼭짓점 추가
        _pts.append((x, y))
        print(f"  꼭짓점 ({x}, {y})  총 {len(_pts)}개")
    elif event == cv2.EVENT_RBUTTONDOWN:     # 우클릭 : Zone 완성
        _finish_zone()


# 함수 이름 : draw_zones()
# 기능      : zones 목록에 있는 모든 완성된 Zone을
#             반투명 채우기와 외곽선으로 프레임에 그린다.
# 파라미터  : np.ndarray frame -> 그림을 그릴 대상 프레임 (BGR 이미지)
# 반환값    : 없음
def draw_zones(frame: np.ndarray):
    for z in zones:
        poly = np.array(z["pts"], dtype=np.int32)

        # 반투명 채우기 : 원본 프레임과 채운 오버레이를 25:75 비율로 합성한다.
        ov = frame.copy()
        cv2.fillPoly(ov, [poly], z["color"])
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)

        cv2.polylines(frame, [poly], True, z["color"], 2)  # 외곽선 그리기

        # Zone 이름을 다각형 중심에 표시한다.
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, z["name"], (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)


# 함수 이름 : draw_current()
# 기능      : 현재 그리는 중인 꼭짓점들을 선으로 연결하여 표시하고,
#             마우스 위치까지 이어지는 미리보기 선과 십자선 커서를 그린다.
# 파라미터  : np.ndarray frame -> 그림을 그릴 대상 프레임 (BGR 이미지)
# 반환값    : 없음
def draw_current(frame: np.ndarray):
    color = _color()

    if _pts:
        poly = np.array(_pts, dtype=np.int32)
        cv2.polylines(frame, [poly], False, color, 2)  # 찍은 꼭짓점 연결선

        for p in _pts:
            cv2.circle(frame, p, 5, color, -1)         # 각 꼭짓점에 점 표시

        cv2.line(frame, _pts[-1], _mouse, color, 1)    # 마지막 점 → 마우스 실선

        if len(_pts) >= 3:                              # 3점 이상이면 닫힘 미리보기
            _dashed(frame, _mouse, _pts[0], color)

    # 마우스 위치에 십자선 커서를 그린다.
    mx, my = _mouse
    cv2.line(frame, (mx - 14, my), (mx + 14, my), color, 1)
    cv2.line(frame, (mx, my - 14), (mx, my + 14), color, 1)


# 함수 이름 : draw_hud()
# 기능      : 현재 모드(재생/그리기)에 맞는 조작 안내 텍스트를
#             화면 하단에 표시한다.
# 파라미터  : np.ndarray frame -> 텍스트를 그릴 대상 프레임 (BGR 이미지)
# 반환값    : 없음
def draw_hud(frame: np.ndarray):
    h = frame.shape[0]  # 프레임 높이 (텍스트 y 좌표 계산용)

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

    # 두 줄의 안내 텍스트를 화면 하단에 순서대로 출력한다.
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, h - 36 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


# 함수 이름 : _dashed()
# 기능      : 두 점 사이를 일정 간격의 점선으로 그린다.
# 파라미터  : np.ndarray img   -> 그림을 그릴 대상 이미지
#             tuple      p1    -> 시작점 (x, y)
#             tuple      p2    -> 끝점   (x, y)
#             tuple      color -> 선 색상 (B, G, R)
#             int        gap   -> 점선 한 칸의 길이 (픽셀, 기본값 8)
# 반환값    : 없음
def _dashed(img, p1, p2, color, gap=8):
    x1, y1 = p1
    x2, y2 = p2
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # 두 점 사이 거리 계산

    if dist == 0:
        return

    n = max(int(dist / gap), 1)  # 점선 칸 수 계산

    # 짝수 인덱스 구간만 선을 그려 점선 효과를 낸다.
    for i in range(0, n, 2):
        t1, t2 = i / n, min((i + 1) / n, 1.0)
        cv2.line(img,
                 (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1)),
                 (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2)),
                 color, 1, cv2.LINE_AA)


# 함수 이름 : save_zones()
# 기능      : 현재 zones 목록을 JSON 파일로 저장한다.
# 파라미터  : str path -> 저장할 파일 경로 (기본값 ZONE_FILE)
# 반환값    : 없음
def save_zones(path: str = ZONE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "zones": [
                    {"name": z["name"], "pts": z["pts"], "color": list(z["color"])}
                    for z in zones
                ],
            },
            f,
            indent=2,
        )
    print(f"[저장] {len(zones)}개 Zone → {path}")


# 함수 이름 : load_zones()
# 기능      : JSON 파일에서 Zone 목록을 불러와 전역 zones 변수에 저장한다.
# 파라미터  : str path -> 불러올 파일 경로 (기본값 ZONE_FILE)
# 반환값    : 없음
def load_zones(path: str = ZONE_FILE):
    global zones
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    zones = [
        {
            "name":  z["name"],
            "pts":   [tuple(p) for p in z["pts"]],
            "color": tuple(z["color"]),
        }
        for z in data["zones"]
    ]
    print(f"[불러오기] {len(zones)}개 Zone ← {path}")


# 함수 이름 : main()
# 기능      : 프로그램 진입점.
#             YOLO 모델을 로드하고, 영상을 재생하면서
#             스페이스바로 일시정지 후 Zone을 설정하고,
#             설정된 Zone 내 킥보드 침범을 감지하여 알림을 출력한다.
# 파라미터  : 없음
# 반환값    : 없음
def main():
    global _draw_mode, _base_frame, _pts, _cidx, _zone_num

    import os, time

    # YOLO 모델을 로드한다.
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully!")

    # 저장된 Zone 파일이 있으면 불러올지 사용자에게 묻는다.
    if os.path.exists(ZONE_FILE):
        ans = input(f"저장된 Zone({ZONE_FILE})을 불러올까요? [y/n]: ").strip().lower()
        if ans == 'y':
            load_zones()

    # 알림 쿨다운 상태를 관리하는 딕셔너리를 초기화한다.
    last_alert: dict = {}

    # 함수 이름 : should_alert()
    # 기능      : 동일 Zone + 동일 객체에 대해 쿨다운이 지났는지 확인한다.
    # 파라미터  : str zone_name -> 침범된 Zone 이름
    #             int tid       -> 객체 추적 ID
    # 반환값    : bool -> True 이면 알림 발생 가능, False 이면 쿨다운 중
    def should_alert(zone_name, tid):
        key = (zone_name, tid)
        if time.time() - last_alert.get(key, 0) >= COOLDOWN:
            last_alert[key] = time.time()  # 마지막 알림 시각 갱신
            return True
        return False

    # 영상 캡처 객체와 결과 저장용 VideoWriter를 초기화한다.
    cap    = cv2.VideoCapture(SOURCE)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_res  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = (
        cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h_res))
        if SAVE_PATH else None
    )

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)  # 마우스 콜백 등록

    print("\n[실행 중]  SPACE=일시정지/구역설정  s=저장  q=종료\n")

    while True:

        # 그리기 모드 : 영상 일시정지 후 Zone 편집
        if _draw_mode:
            # 일시정지된 배경 프레임에 Zone과 현재 그리는 선을 합성하여 표시한다.
            frame = _base_frame.copy()
            draw_zones(frame)
            draw_current(frame)
            draw_hud(frame)
            cv2.imshow(WIN, frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):                   # 스페이스 : 재생 재개
                _draw_mode = False
                print("[재생 재개]")

            elif key == ord('n'):                 # n : 새 Zone 시작
                if len(_pts) >= 3:
                    _finish_zone()
                else:
                    _pts = []
                    _cidx += 1

            elif key == ord('z'):                 # z : 마지막 꼭짓점 되돌리기
                if _pts:
                    print(f"  되돌리기: {_pts.pop()}")

            elif key == ord('c'):                 # c : 현재 그리던 Zone 초기화
                _pts = []

            elif key == ord('r'):                 # r : 완성된 Zone 포함 전체 초기화
                _pts = []
                zones.clear()
                _cidx = 0
                _zone_num = 1
                print("[전체 초기화]")

            elif key == ord('s'):                 # s : 현재 Zone 완성 후 파일 저장
                if len(_pts) >= 3:
                    _finish_zone()
                save_zones()

            elif key in (ord('q'), 27):           # q / ESC : 종료
                break

            continue   # 재생 루프로 넘어가지 않고 그리기 루프 반복

        #재생 모드 : 프레임 읽기 → 추적 → 침범 감지
        ret, frame = cap.read()
        if not ret:   # 영상 끝에 도달하면 루프 종료
            break

        # YOLO 로 객체를 추적한다. (ByteTrack 내장)
        results = model.track(frame, persist=True, conf=CONF, verbose=False)

        draw_zones(frame)   # 설정된 Zone 오버레이 렌더링

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

                # 객체 레이블(클래스명, 추적ID, 신뢰도)을 바운딩 박스에 표시한다.
                ann.box_label(
                    (x1, y1, x2, y2),
                    f"{model.names[cid]} #{tid} {conf:.2f}",
                    color=yolo_colors(cid, bgr=True),
                )

                # 바운딩 박스 하단 중심(발 위치)이 각 Zone 내부인지 확인한다.
                for z in zones:
                    poly    = np.array(z["pts"], dtype=np.int32)
                    cx, cy  = int((x1 + x2) / 2), int(y2)  # 발 위치 좌표

                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        # 침범 시 빨간 박스와 경고 텍스트를 프레임에 그린다.
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame,
                            f"! {model.names[cid]} IN {z['name']}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                        )

                        # 쿨다운이 지난 경우에만 터미널에 알림을 출력한다.
                        if should_alert(z["name"], int(tid)):
                            print(
                                f"[ALERT] {datetime.now().strftime('%H:%M:%S')} | "
                                f"{z['name']} | #{tid} {model.names[cid]} ({conf:.2f})"
                            )

        draw_hud(frame)   # 조작 안내 텍스트 렌더링

        if writer:
            writer.write(frame)   # 결과 프레임을 영상 파일에 저장한다.

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):                       # 스페이스 : 현재 프레임에서 일시정지
            _draw_mode  = True
            _base_frame = frame.copy()
            print("[일시정지]  마우스로 Zone을 그리세요  SPACE=재생재개")
        elif key == ord('s'):                     # s : Zone 파일 저장
            save_zones()
        elif key in (ord('q'), 27):               # q / ESC : 종료
            break

    # 자원 해제
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[완료]")


if __name__ == "__main__":
    main()