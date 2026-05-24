import queue
import threading
import time
import cv2
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
import config


# 클래스명        : DetectionLoop
# 기능           : 영상 캡처 및 YOLO 추론 루프 전담 (SRP 분리)
# 내장 함수 목록  : __init__()                    - 모델, 렌더러, 판정기 초기화
#                  _capture()                    - 별도 스레드에서 프레임 캡처 및 큐 공급
#                  _sync_helmet_ids()            - 탑승자 박스 내부 헬멧의 ID를 탑승자 ID로 교체
#                  _apply_smoothing_and_filter() - EMA 스무딩 및 최소 확정 프레임 필터링
#                  run()                         - 큐에서 프레임을 받아 YOLO 추론 및 인코딩 루프 실행
class DetectionLoop:

    # 함수 이름 : __init__()
    # 기능      : DetectionLoop 객체를 초기화한다.
    # 파라미터  : YOLO            model    -> 이미 로드된 YOLO 모델 인스턴스
    #             ZoneRenderer    renderer -> Zone 렌더링 담당 인스턴스
    #             DecideViolation decider  -> 위반 판정 담당 인스턴스
    # 반환값    : 없음
    def __init__(self, model, renderer, decider):
        self.model    = model
        self.renderer = renderer
        self.decider  = decider
        self._q       = queue.Queue(maxsize=1)  # 추론이 느릴 때 오래된 프레임을 버린다.
        self._ema     = {}                      # {(track_id, label): 스무딩된 박스 좌표}
        self._confirm = {}                      # {track_id: 연속 감지 프레임 수}

    # 함수 이름 : _capture()
    # 기능      : 별도 스레드에서 영상을 읽어 큐에 최신 프레임만 유지한다.
    #             큐가 가득 찬 경우 오래된 프레임을 꺼내 버리고 새 프레임으로 교체한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def _capture(self):
        cap = cv2.VideoCapture(config.SOURCE)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30    # 영상 FPS를 읽어 프레임 간격을 계산한다.
        delay = 1.0 / fps
        while True:
            ret, frame = cap.read()
            if not ret:                         # 영상 끝에 도달하면 처음부터 재생한다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                try:
                    self._q.get_nowait()        # 오래된 프레임 제거
                except queue.Empty:
                    pass
                self._q.put_nowait(frame)       # 최신 프레임으로 교체
            time.sleep(delay)                   # 실시간 재생 속도 유지
        cap.release()

    # 함수 이름 : _sync_helmet_ids()
    # 기능      : 탑승자 박스 내부에 중심이 있는 헬멧의 트랙 ID를 해당 탑승자 ID로 교체한다.
    #             헬멧이 탑승자의 확정된 트랙 ID를 물려받아 MIN_CONFIRM_FRAMES 대기 없이 즉시 표시된다.
    # 파라미터  : list boxes  -> 바운딩 박스 목록 [(x1,y1,x2,y2), ...]
    #             list labels -> 레이블 목록
    #             list ids    -> 트랙 ID 목록
    # 반환값    : list -> ID 동기화가 완료된 트랙 ID 목록
    def _sync_helmet_ids(self, boxes, labels, ids):
        RIDER_LABELS  = {"person_with_kickboard", "2-person_with_kickboard"}
        HELMET_LABELS = {"helmet_O", "helmet_X"}

        # 탑승자 박스와 ID를 미리 수집한다.
        riders = [(boxes[i], ids[i]) for i, l in enumerate(labels) if l in RIDER_LABELS]

        synced = list(ids)
        for i, label in enumerate(labels):
            if label not in HELMET_LABELS:
                continue
            hx1, hy1, hx2, hy2 = boxes[i]
            hcx = (hx1 + hx2) / 2      # 헬멧 박스 중심 X
            hcy = (hy1 + hy2) / 2      # 헬멧 박스 중심 Y
            for (rx1, ry1, rx2, ry2), rtid in riders:
                if rx1 <= hcx <= rx2 and ry1 <= hcy <= ry2:
                    synced[i] = rtid    # 탑승자 ID 로 교체
                    break
        return synced

    # 함수 이름 : _apply_smoothing_and_filter()
    # 기능      : EMA 스무딩으로 박스 좌표를 안정화하고, MIN_CONFIRM_FRAMES 미만 트랙을 필터링한다.
    #             사라진 트랙 ID 는 내부 딕셔너리에서 제거한다.
    # 파라미터  : list boxes   -> 바운딩 박스 목록 [(x1,y1,x2,y2), ...]
    #             list labels  -> 레이블 목록
    #             list confs   -> 신뢰도 목록
    #             list ids     -> 트랙 ID 목록
    #             list cls_ids -> 클래스 인덱스 목록
    # 반환값    : tuple -> (boxes, labels, confs, ids, cls_ids) 필터링 및 스무딩 적용 후 목록
    def _apply_smoothing_and_filter(self, boxes, labels, confs, ids, cls_ids):
        current_tids     = set(ids)
        current_ema_keys = set(zip(ids, labels))

        for tid in set(self._confirm) - current_tids:      # 사라진 트랙 ID 정리
            self._confirm.pop(tid, None)
        for key in set(self._ema) - current_ema_keys:      # 사라진 (ID, 레이블) EMA 항목 정리
            self._ema.pop(key, None)

        seen_tids = set()
        filtered  = []
        for i, tid in enumerate(ids):
            if tid not in seen_tids:                        # 동일 ID 중복 카운트 방지
                self._confirm[tid] = self._confirm.get(tid, 0) + 1
                seen_tids.add(tid)

            ema_key = (tid, labels[i])                      # 레이블별 EMA 를 독립적으로 관리한다.
            box = boxes[i]
            if ema_key in self._ema:
                a   = config.EMA_ALPHA
                box = tuple(int(a * c + (1 - a) * p) for c, p in zip(box, self._ema[ema_key]))
            self._ema[ema_key] = box

            if self._confirm[tid] >= config.MIN_CONFIRM_FRAMES:
                filtered.append((box, labels[i], confs[i], tid, cls_ids[i]))

        if not filtered:
            return [], [], [], [], []
        f_boxes, f_labels, f_confs, f_ids, f_cls = zip(*filtered)
        return list(f_boxes), list(f_labels), list(f_confs), list(f_ids), list(f_cls)

    # 함수 이름 : run()
    # 기능      : 캡처 스레드를 시작하고 큐에서 프레임을 받아 YOLO 로 추적하면서 decider.check() 를 호출한다.
    #             처리된 프레임을 JPEG 로 인코딩하여 config.latest_frame 에 저장한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def run(self):
        threading.Thread(target=self._capture, daemon=True).start()
        print("\n[감지 루프 시작]\n")

        while True:
            frame = self._q.get()               # 최신 프레임이 올 때까지 대기
            frame = cv2.resize(frame, config.INFER_SIZE)  # 추론 전 해상도 축소
            # YOLO 로 객체를 추적한다.
            results = self.model.track(
                frame, persist=True, conf=config.CONF, verbose=False,
                tracker="bytetrack.yaml", vid_stride=1
            )

            self.renderer.draw_zones(frame)     # Zone 오버레이 렌더링

            if results[0].boxes is not None:
                raw_boxes = results[0].boxes.xyxy.cpu().numpy()
                cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                confs     = results[0].boxes.conf.cpu().numpy().tolist()
                ids       = (
                    results[0].boxes.id.cpu().numpy().astype(int).tolist()
                    if results[0].boxes.id is not None
                    else list(range(len(raw_boxes)))
                )
                labels = [self.model.names[c] for c in cls_ids]
                boxes  = [tuple(map(int, b)) for b in raw_boxes]

                # 헬멧 ID 를 탑승자 ID 로 동기화한다.
                ids = self._sync_helmet_ids(boxes, labels, ids)

                # EMA 스무딩 및 최소 확정 프레임 필터링을 적용한다.
                boxes, labels, confs, ids, cls_ids = self._apply_smoothing_and_filter(
                    boxes, labels, confs, ids, cls_ids
                )

                if boxes:
                    ann = Annotator(frame, line_width=2)
                    for box, label, conf, tid, cid in zip(boxes, labels, confs, ids, cls_ids):
                        ann.box_label(box, f"{label} #{tid} {conf:.2f}",
                                      color=yolo_colors(cid, bgr=True))

                    # 위반 판정을 수행한다.
                    self.decider.check(frame, boxes, labels, confs, ids)

            # 프레임을 JPEG 로 인코딩하여 스트리밍용 전역 변수에 저장한다.
            _, buf = cv2.imencode(".jpg", frame, config.ENCODE_PARAMS)
            config.latest_frame = buf.tobytes()
