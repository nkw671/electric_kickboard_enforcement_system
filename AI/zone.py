import cv2
import json
import numpy as np
from datetime import datetime
from config import ZONE_FILE


# 클래스명        : ZoneDrawer
# 기능           : Zone 데이터 관리 전담 (렌더링 책임은 ZoneRenderer 로 분리)
# 내장 함수 목록  : __init__()    - Zone 상태 변수 초기화
#                  _color()      - 현재 색상 인덱스에 해당하는 BGR 색상 튜플 반환
#                  finish_zone() - 현재 꼭짓점으로 Zone 완성하여 zones 목록에 추가
#                  save()        - 현재 zones 목록을 JSON 파일로 저장
#                  load()        - JSON 파일에서 Zone 목록을 불러와 zones 에 저장
#                  set_zones()   - API 에서 받은 좌표 목록으로 zones 를 교체한다
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


# 클래스명        : ZoneRenderer
# 기능           : Zone 렌더링 전담 (SRP 분리)
# 내장 함수 목록  : __init__()     - ZoneDrawer 참조 초기화
#                  draw_zones()   - 완성된 모든 Zone을 반투명 채우기와 외곽선으로 프레임에 표시
#                  _draw_dashed() - 두 점 사이를 일정 간격의 점선으로 표시
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
