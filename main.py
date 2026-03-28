import cv2
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
from PIL import ImageFont, ImageDraw, Image
# 추가
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────

SOURCE     = "1.mp4"       # 입력 영상 경로 (웹캠은 0)
MODEL_PATH = "src/best_v3.pt"  # YOLO 모델 가중치 경로
CONF       = 0.5           # 객체 감지 최소 신뢰도
COOLDOWN   = 3.0           # 동일 객체 재알림 최소 간격 (초)
SAVE_PATH  = None  # 결과 영상 저장 경로 (None 이면 저장 안 함)
ZONE_FILE  = "zones.json"  # Zone 좌표 저장/불러오기 파일 경로
latest_frame: bytes = b""   # 감지 루프에서 매 프레임 저장

WIN = "Kickboard Zone Monitor"  # OpenCV 창 이름



# 클래스명        : ZoneDrawer
# 기능           : 영상 화면 안에서 구역을 설정한다.
# 내장 함수 목록   :  __init__()        - Zone 그리기에 필요한 모든 상태 변수 초기화
#                   _color()          - 현재 색상 인덱스에 해당하는 BGR 색상 튜플 반환
#                   finish_zone()     - 현재 꼭짓점으로 Zone을 완성하여 zones 목록에 추가
#                   on_mouse()        - 마우스 이벤트 처리
#                   enter_draw_mode() - 현재 프레임을 배경으로 저장하고 그리기 모드로 전환
#                   exit_draw_mode()  - 그리기 모드를 종료하고 재생 모드로 전환
#                   handle_key()      - 그리기 모드에서 키 입력을 처리하고 종료 여부 반환
#                   render()          - 배경 프레임에 Zone, 미리보기 선, HUD를 합성하여 반환
#                   draw_zones()      - 완성된 모든 Zone을 반투명 채우기와 외곽선으로 프레임에 표시
#                   _draw_current()   - 현재 그리는 꼭짓점, 미리보기 선, 십자선 커서를 프레임에 표시
#                   _draw_hud()       - 현재 모드에 맞는 조작 안내 텍스트를 화면 하단에 표시
#                   _draw_dashed()    - 두 점 사이를 일정 간격의 점선으로 표시
#                   save()            - 현재 zones 목록을 JSON 파일로 저장
#                   load()            - JSON 파일에서 Zone 목록을 불러와 zones에 저장
#                   in_zone()         - 바운딩 박스 하단 중심이 특정 Zone 내부인지 확인(추후 위반사항 판별 기능 구현 시 분리할 예정)

class ZoneDrawer:
    """
    영상 프레임 위에서 마우스로 감시 Zone을 그리고 관리하는 클래스.

    조작법:
      좌클릭       꼭짓점 추가
      우클릭       현재 Zone 완성 (3점 이상)
      n            새 Zone 시작
      z            마지막 꼭짓점 되돌리기
      c            현재 Zone 초기화
      r            전체 초기화
      s            Zone 파일 저장
    """

    # Zone 색상 팔레트 (BGR 순서)
    COLORS = [
        (0,   0,   255),
        (0,  165,  255),
        (0,  255,    0),
        (255, 100,   0),
        (255,   0,  200),
    ]

    # 함수 이름 : __init__()
    # 기능      : ZoneDrawer 객체를 초기화한다.
    #             Zone 목록, 꼭짓점, 마우스 위치 등 그리기에 필요한
    #             모든 상태 변수를 초기화한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def __init__(self):
        self.zones      = []       # 완성된 Zone 딕셔너리 목록
        self._pts       = []       # 현재 그리는 중인 꼭짓점 좌표 목록
        self._mouse     = (0, 0)   # 현재 마우스 커서 위치
        self._cidx      = 0        # 현재 선택된 색상 인덱스
        self._zone_num  = 1        # 다음 Zone 이름에 붙을 번호
        self.draw_mode  = False    # True 이면 일시정지 후 Zone 그리기 모드
        self._base_frame = None    # 일시정지 시 캡처한 배경 프레임
        self._font = self._load_font(18)

    # 함수 이름 : _load_font()
    # 기능      : 시스템에서 사용 가능한 한글 폰트를 순서대로 탐색하여 로드한다.
    #             찾지 못하면 PIL 기본 폰트를 반환한다.
    # 파라미터  : int size -> 폰트 크기 (픽셀)
    # 반환값    : ImageFont 객체
    @staticmethod
    def _load_font(size: int):
        candidates = [
            "malgun.ttf",  # Windows 맑은 고딕
            "C:/Windows/Fonts/malgun.ttf",  # Windows 절대 경로
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux 나눔고딕
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    # 함수 이름 : _color()
    # 기능      : 현재 색상 인덱스에 해당하는 BGR 색상 튜플을 반환한다.
    # 파라미터  : 없음
    # 반환값    : tuple -> COLORS 팔레트에서 선택된 (B, G, R) 색상 튜플
    def _color(self):
        return self.COLORS[self._cidx % len(self.COLORS)]

    # 함수 이름 : finish_zone()
    # 기능      : 현재까지 찍은 꼭짓점으로 Zone을 완성하여 zones 목록에 추가한다.
    #             꼭짓점이 3개 미만이면 경고 메시지를 출력하고 종료한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def finish_zone(self):
        if len(self._pts) < 3:                  # 다각형 성립 최소 조건 검사
            print("[경고] 최소 3개 꼭짓점 필요")
            return

        # Zone 이름을 생성하고 zones 목록에 추가한다.
        name = f"Zone-{self._zone_num}"
        self.zones.append({"name": name, "pts": list(self._pts), "color": self._color()})
        print(f"[✓] {name} 완성 ({len(self._pts)}개 꼭짓점)")

        self._pts      = []               # 꼭짓점 목록 초기화
        self._cidx    += 1                # 다음 Zone 은 다른 색상 사용
        self._zone_num += 1               # Zone 번호 증가

    # 함수 이름 : on_mouse()
    # 기능      : OpenCV 마우스 이벤트를 처리한다.
    #             draw_mode 가 True 일 때만 동작하며,
    #             좌클릭이면 꼭짓점을 추가하고 우클릭이면 Zone 을 완성한다.
    # 파라미터  : int    event  -> OpenCV 마우스 이벤트 상수
    #             int    x      -> 마우스 클릭 x 좌표 (픽셀)
    #             int    y      -> 마우스 클릭 y 좌표 (픽셀)
    #             int    flags  -> 추가 이벤트 플래그 (미사용)
    #             object _      -> 사용자 데이터 (미사용)
    # 반환값    : 없음
    def on_mouse(self, event, x, y, flags, _):
        if not self.draw_mode:          # 재생 모드에서는 마우스 입력 무시
            return

        self._mouse = (x, y)            # 현재 마우스 위치 갱신

        if event == cv2.EVENT_LBUTTONDOWN:      # 좌클릭 : 꼭짓점 추가
            self._pts.append((x, y))
            print(f"  꼭짓점 ({x}, {y})  총 {len(self._pts)}개")
        elif event == cv2.EVENT_RBUTTONDOWN:    # 우클릭 : Zone 완성
            self.finish_zone()

    # 함수 이름 : enter_draw_mode()
    # 기능      : 현재 프레임을 배경으로 저장하고 그리기 모드로 전환한다.
    # 파라미터  : np.ndarray frame -> 일시정지할 현재 영상 프레임
    # 반환값    : 없음
    def enter_draw_mode(self, frame: np.ndarray):
        self.draw_mode   = True
        self._base_frame = frame.copy()  # 일시정지 프레임 저장
        print("[일시정지]  마우스로 Zone을 그리세요  SPACE=재생재개")

    # 함수 이름 : exit_draw_mode()
    # 기능      : 그리기 모드를 종료하고 재생 모드로 전환한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def exit_draw_mode(self):
        self.draw_mode = False
        print("[재생 재개]")

    # 함수 이름 : handle_key()
    # 기능      : 그리기 모드에서 키 입력을 처리한다.
    #             각 키에 맞는 Zone 편집 동작을 수행하고,
    #             종료 여부를 bool 값으로 반환한다.
    # 파라미터  : int key -> cv2.waitKey() 로 읽은 키 코드
    # 반환값    : bool -> True 이면 프로그램 종료 요청
    def handle_key(self, key: int) -> bool:
        if key == ord(' '):                     # 스페이스 : 재생 재개
            self.exit_draw_mode()

        elif key == ord('n'):                   # n : 새 Zone 시작
            if len(self._pts) >= 3:
                self.finish_zone()
            else:
                self._pts  = []
                self._cidx += 1

        elif key == ord('z'):                   # z : 마지막 꼭짓점 되돌리기
            if self._pts:
                print(f"  되돌리기: {self._pts.pop()}")

        elif key == ord('c'):                   # c : 현재 그리던 Zone 초기화
            self._pts = []

        elif key == ord('r'):                   # r : 완성된 Zone 포함 전체 초기화
            self._pts      = []
            self.zones.clear()
            self._cidx     = 0
            self._zone_num = 1
            print("[전체 초기화]")

        elif key == ord('s'):                   # s : 현재 Zone 완성 후 파일 저장
            if len(self._pts) >= 3:
                self.finish_zone()
            self.save(ZONE_FILE)

        elif key in (ord('q'), 27):             # q / ESC : 종료
            return True

        return False                            # 종료 요청 없음

    # 함수 이름 : render()
    # 기능      : 일시정지된 배경 프레임에 완성된 Zone, 현재 그리는 선,
    #             HUD 안내 텍스트를 합성하여 반환한다.
    # 파라미터  : 없음
    # 반환값    : np.ndarray -> 렌더링이 완료된 프레임 (BGR 이미지)
    def render(self) -> np.ndarray:
        frame = self._base_frame.copy()
        self.draw_zones(frame)      # 완성된 Zone 오버레이
        self._draw_current(frame)   # 현재 그리는 선 + 커서
        self._draw_hud(frame)       # 조작 안내 텍스트
        return frame

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

            cv2.polylines(frame, [poly], True, z["color"], 2)  # 외곽선 그리기

            # Zone 이름을 다각형 중심에 표시한다.
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, z["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, z["color"], 2)

    # 함수 이름 : _draw_current()
    # 기능      : 현재 그리는 중인 꼭짓점들을 선으로 연결하여 표시하고,
    #             마우스 위치까지 이어지는 미리보기 선과 십자선 커서를 그린다.
    # 파라미터  : np.ndarray frame -> 그림을 그릴 대상 프레임 (BGR 이미지)
    # 반환값    : 없음
    def _draw_current(self, frame: np.ndarray):
        color = self._color()

        if self._pts:
            poly = np.array(self._pts, dtype=np.int32)
            cv2.polylines(frame, [poly], False, color, 2)  # 찍은 꼭짓점 연결선

            for p in self._pts:
                cv2.circle(frame, p, 5, color, -1)         # 각 꼭짓점에 점 표시

            cv2.line(frame, self._pts[-1], self._mouse, color, 1)  # 마지막 점 → 마우스 실선

            if len(self._pts) >= 3:                         # 3점 이상이면 닫힘 미리보기
                self._draw_dashed(frame, self._mouse, self._pts[0], color)

        # 마우스 위치에 십자선 커서를 그린다.
        mx, my = self._mouse
        cv2.line(frame, (mx - 14, my), (mx + 14, my), color, 1)
        cv2.line(frame, (mx, my - 14), (mx, my + 14), color, 1)

        # 함수 이름 : _put_text_ko()
        # 기능      : PIL을 사용하여 OpenCV 프레임에 한글 텍스트를 그린다.
        #             OpenCV 는 한글을 지원하지 않으므로 PIL로 변환 후 다시 BGR로 복원한다.
        # 파라미터  : np.ndarray frame -> 텍스트를 그릴 대상 프레임 (BGR 이미지)
        #             str        text  -> 출력할 문자열 (한글 포함 가능)
        #             tuple      pos   -> 텍스트 좌상단 좌표 (x, y)
        #             int        size  -> 폰트 크기 (픽셀)
        #             tuple      color -> 텍스트 색상 (R, G, B)
        # 반환값    : 없음

    def _put_korean_text(self, frame: np.ndarray, text: str, pos: tuple,
                         color: tuple = (220, 220, 220)):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR → RGB 변환
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=self._font, fill=color)  # 텍스트 그리기
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB → BGR 복원
        np.copyto(frame, result)  # 원본 프레임에 반영

        # 함수 이름 : _draw_hud()
        # 기능      : 현재 모드(재생/그리기)에 맞는 조작 안내 텍스트를
        #             화면 하단에 한글로 표시한다.
        # 파라미터  : np.ndarray frame -> 텍스트를 그릴 대상 프레임 (BGR 이미지)
        # 반환값    : 없음

    def _draw_hud(self, frame: np.ndarray):
        h = frame.shape[0]  # 프레임 높이 (텍스트 y 좌표 계산용)

        if self.draw_mode:
            lines = [
                f"[일시정지 - 구역 그리기]  완성: {len(self.zones)}개  현재 꼭짓점: {len(self._pts)}개",
                "좌클릭=점추가  우클릭=Zone완성  n=새구역  z=되돌리기  SPACE=재생재개  q=종료",
            ]
        else:
            lines = [
                f"[재생 중]  설정된 Zone: {len(self.zones)}개",
                "SPACE=일시정지(구역설정)  s=Zone저장  q=종료",
            ]

        # 두 줄의 안내 텍스트를 화면 하단에 순서대로 출력한다.
        for i, text in enumerate(lines):
            self._put_korean_text(frame, text, pos=(10, h - 50 + i * 24))

    # 함수 이름 : _draw_dashed()
    # 기능      : 두 점 사이를 일정 간격의 점선으로 그린다.
    # 파라미터  : np.ndarray img   -> 그림을 그릴 대상 이미지
    #             tuple      p1    -> 시작점 (x, y)
    #             tuple      p2    -> 끝점   (x, y)
    #             tuple      color -> 선 색상 (B, G, R)
    #             int        gap   -> 점선 한 칸의 길이 (픽셀, 기본값 8)
    # 반환값    : 없음
    def _draw_dashed(self, img, p1, p2, color, gap=8):
        x1, y1 = p1
        x2, y2 = p2
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5   # 두 점 사이 거리 계산

        if dist == 0:
            return

        n = max(int(dist / gap), 1)   # 점선 칸 수 계산

        # 짝수 인덱스 구간만 선을 그려 점선 효과를 낸다.
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
        print(f"[저장] {len(self.zones)}개 Zone → {path}")

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
        print(f"[불러오기] {len(self.zones)}개 Zone ← {path}")

    # 함수 이름 : in_zone()
    # 기능      : 바운딩 박스의 하단 중심(발 위치)이 특정 Zone 내부인지 확인한다.
    # 파라미터  : dict zone       -> 확인할 Zone 딕셔너리
    #             int  x1, y1    -> 바운딩 박스 좌상단 좌표 (픽셀)
    #             int  x2, y2    -> 바운딩 박스 우하단 좌표 (픽셀)
    # 반환값    : bool -> True 이면 발 위치가 Zone 내부
    def in_zone(self, zone: dict, x1, y1, x2, y2) -> bool:
        poly   = np.array(zone["pts"], dtype=np.int32)
        cx, cy = int((x1 + x2) / 2), int(y2)   # 발 위치 좌표
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0



# 함수 이름 : main()
# 기능      : 프로그램 진입점.
#             YOLO 모델을 로드하고, 영상을 재생하면서
#             스페이스바로 일시정지 후 Zone을 설정하고,
#             설정된 Zone 내 킥보드 침범을 감지하여 알림을 출력한다.
# 파라미터  : 없음
# 반환값    : 없음
def main():
    import os, time

    # YOLO 모델을 로드한다.
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully!")

    # ZoneDrawer 객체를 생성한다.
    drawer = ZoneDrawer()

    # 저장된 Zone 파일이 있으면 불러올지 사용자에게 묻는다.
    if os.path.exists(ZONE_FILE):
        ans = input(f"저장된 Zone({ZONE_FILE})을 불러올까요? [y/n]: ").strip().lower()
        if ans == 'y':
            drawer.load(ZONE_FILE)

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
            last_alert[key] = time.time()   # 마지막 알림 시각 갱신
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
    cv2.setMouseCallback(WIN, drawer.on_mouse)  # 마우스 콜백을 drawer 에 연결

    print("\n[실행 중]  SPACE=일시정지/구역설정  s=저장  q=종료\n")

    while True:

        # ── 그리기 모드 : 영상 일시정지 후 Zone 편집 ──────────────
        if drawer.draw_mode:
            cv2.imshow(WIN, drawer.render())    # 그리기 결과를 화면에 표시한다.

            key = cv2.waitKey(30) & 0xFF
            if drawer.handle_key(key):          # 종료 키 입력 시 루프 탈출
                break
            continue                            # 재생 루프로 넘어가지 않고 반복

        # ── 재생 모드 : 프레임 읽기 → 추적 → 침범 감지 ───────────
        ret, frame = cap.read()
        if not ret:                             # 영상 끝에 도달하면 루프 종료
            break

        # YOLO 로 객체를 추적한다. (ByteTrack 내장)
        results = model.track(frame, persist=True, conf=CONF, verbose=False)

        drawer.draw_zones(frame)   # 설정된 Zone 오버레이 렌더링

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

                # 각 Zone 에 대해 침범 여부를 확인한다.
                for z in drawer.zones:
                    if drawer.in_zone(z, x1, y1, x2, y2):
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

        drawer._draw_hud(frame)   # 조작 안내 텍스트 렌더링

        if writer:
            writer.write(frame)   # 결과 프레임을 영상 파일에 저장한다.

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = buf.tobytes()  # 최신 프레임을 전역 변수에 저장한다.
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):                     # 스페이스 : 현재 프레임에서 일시정지
            drawer.enter_draw_mode(frame)
        elif key == ord('s'):                   # s : Zone 파일 저장
            drawer.save(ZONE_FILE)
        elif key in (ord('q'), 27):             # q / ESC : 종료
            break

    # 자원 해제
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[완료]")

# ──────────────────────────────────────────────
# FastAPI 스트리밍 서버
# ──────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 함수 이름 : video_stream()
# 기능      : latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다.
# 반환값    : StreamingResponse (multipart/x-mixed-replace)
@app.get("/video/stream")
async def video_stream():
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

if __name__ == "__main__":
    import uvicorn

    # 감지 루프를 별도 스레드에서 실행한다. (OpenCV 창 + 감지 유지)
    Thread(target=main, daemon=True).start()

    # FastAPI 서버를 실행한다.
    uvicorn.run(app, host="0.0.0.0", port=8000)