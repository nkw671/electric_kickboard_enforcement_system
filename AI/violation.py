import cv2
import time
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from config import COOLDOWN, VIOLATION_DIR
import os


# 클래스명        : ViolationStrategy
# 기능           : 위반 판정 전략 추상 인터페이스 (OCP 적용)
# 내장 함수 목록  : check() - 위반 여부를 판정하여 위반 유형 문자열을 반환한다 (추상 메서드)
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


# 클래스명        : HelmetViolation
# 기능           : 헬멧 미착용 판정 전략
# 내장 함수 목록  : check() - helmet_X 박스 중심이 탑승자 박스 내부에 있으면 위반 판정
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


# 클래스명        : SidewalkViolation
# 기능           : 인도주행 판정 전략
# 내장 함수 목록  : __init__() - ZoneDrawer 참조 초기화
#                  check()    - 탑승자 박스 하단 중심이 Zone 안에 있으면 위반 판정
class SidewalkViolation(ViolationStrategy):

    # 함수 이름 : __init__()
    # 기능      : SidewalkViolation 을 초기화한다.
    # 파라미터  : ZoneDrawer zone_drawer -> Zone 정보를 가진 ZoneDrawer 인스턴스
    # 반환값    : 없음
    def __init__(self, zone_drawer):
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


# 클래스명        : DoubleRidingViolation
# 기능           : 다인탑승 판정 전략
# 내장 함수 목록  : check() - 2-person_with_kickboard 레이블이면 다인탑승으로 판정
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


# 클래스명        : DecideViolation
# 기능           : 위반 판정 전담 — 전략 목록 실행, 쿨다운, 렌더링, 콜백 처리
# 내장 함수 목록  : __init__()          - 전략 목록, 쿨다운, 폰트, 콜백 초기화
#                  _load_font()        - 시스템 한글 폰트 탐색 및 로드
#                  _should_alert()     - 동일 객체·위반 유형 쿨다운 여부 확인
#                  _save_frame()       - 위반 프레임을 이미지 파일로 저장
#                  _draw_violations()  - 위반 텍스트를 PIL 변환 1회로 일괄 렌더링
#                  check()             - 매 프레임 전략 목록을 실행하여 위반 종합 판정
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
        self._last_alert  = {}                    # {(track_id, violation_type): 마지막 알림 시각}
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


# 함수 이름 : build_strategies()
# 기능      : 기본 위반 전략 목록을 생성하여 반환한다.
#             새 위반 유형 추가 시 이 함수만 수정하면 된다. (OCP)
# 파라미터  : ZoneDrawer zone_drawer -> SidewalkViolation 에 전달할 Zone 데이터 인스턴스
# 반환값    : list -> ViolationStrategy 인스턴스 목록
def build_strategies(zone_drawer) -> list:
    return [
        HelmetViolation(),
        SidewalkViolation(zone_drawer),
        DoubleRidingViolation(),
    ]
