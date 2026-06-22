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
        self._q         = queue.Queue(maxsize=1)  # 추론이 느릴 때 오래된 프레임을 버린다.
        self._ema       = {}                      # {(track_id, label): 스무딩된 박스 좌표}
        self._confirm   = {}                      # {track_id: 연속 감지 프레임 수}
        self._miss      = {}                      # {(track_id, label): 연속 미검출 프레임 수}
        self._last_meta = {}                      # {(track_id, label): (신뢰도, 클래스 인덱스)}

    # 함수 이름 : _capture()
    # 기능      : 별도 스레드에서 영상을 읽어 큐에 최신 프레임만 유지한다.
    #             영상 타임스탬프와 실제 경과 시간을 비교해 뒤처지면 프레임을 스킵하고
    #             앞서면 대기하여 고FPS 영상에서도 실시간 속도를 유지한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def _capture(self):
        cap = cv2.VideoCapture(config.SOURCE)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self._write_bytetrack_yaml(fps)             # FPS 기반으로 bytetrack.yaml 자동 생성
        start_real = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:                         # 영상 끝에 도달하면 처음부터 재생한다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_real = time.perf_counter()
                continue

            video_ms  = cap.get(cv2.CAP_PROP_POS_MSEC)
            real_ms   = (time.perf_counter() - start_real) * 1000

            if video_ms < real_ms:              # 실시간보다 뒤처진 프레임은 스킵한다.
                continue

            try:
                self._q.put_nowait(frame)
            except queue.Full:
                try:
                    self._q.get_nowait()        # 오래된 프레임 제거
                except queue.Empty:
                    pass
                self._q.put_nowait(frame)       # 최신 프레임으로 교체

            ahead_s = (video_ms - real_ms) / 1000
            if ahead_s > 0:
                time.sleep(ahead_s)             # 실시간보다 앞서면 대기한다.
        cap.release()

    # 함수 이름 : _write_bytetrack_yaml()
    # 기능      : 영상 FPS 에 맞춰 track_buffer 를 계산한 뒤 bytetrack.yaml 을 덮어쓴다.
    #             config.TRACK_BUFFER_SEC 초 동안 트랙을 유지하도록 프레임 수를 환산한다.
    # 파라미터  : float fps -> 영상의 실제 FPS
    # 반환값    : 없음
    def _write_bytetrack_yaml(self, fps: float):
        track_buffer = max(30, int(config.TRACK_BUFFER_SEC * fps))
        content = (
            f"tracker_type: bytetrack\n"
            f"track_high_thresh: 0.25\n"      # 낮춰서 신뢰도 떨어진 검출도 기존 트랙에 재연결
            f"track_low_thresh: 0.10\n"
            f"new_track_thresh: 0.65\n"       # 높여서 어설픈 새 트랙 생성을 억제 (ID 튐 감소)
            f"track_buffer: {track_buffer}          # {fps:.1f}fps × {config.TRACK_BUFFER_SEC}s 자동 계산\n"
            f"match_thresh: 0.85\n"           # 관대한 IoU 매칭 → 빠른 이동·일시 가림에도 ID 유지
            f"fuse_score: True\n"             # 검출 신뢰도를 IoU 매칭에 결합해 연관 안정화
        )
        with open("bytetrack.yaml", "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[ByteTrack] track_buffer={track_buffer} ({fps:.1f}fps × {config.TRACK_BUFFER_SEC}s)")

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
    #             검출이 빠진 확정 트랙은 COAST_MAX_FRAMES 동안 마지막 박스를 유지(coasting)해
    #             간헐적 미검출로 인한 깜박임을 막고, 그 한계를 넘으면 내부 딕셔너리에서 제거한다.
    # 파라미터  : list boxes   -> 바운딩 박스 목록 [(x1,y1,x2,y2), ...]
    #             list labels  -> 레이블 목록
    #             list confs   -> 신뢰도 목록
    #             list ids     -> 트랙 ID 목록
    #             list cls_ids -> 클래스 인덱스 목록
    # 반환값    : tuple -> (boxes, labels, confs, ids, cls_ids) 필터링 및 스무딩 적용 후 목록
    def _apply_smoothing_and_filter(self, boxes, labels, confs, ids, cls_ids):
        current_ema_keys = set(zip(ids, labels))

        seen_tids = set()
        filtered  = []
        # 현재 프레임에 검출된 트랙 처리 : EMA 스무딩 후 확정된 것만 표시 대상에 추가
        for i, tid in enumerate(ids):
            if tid not in seen_tids:                        # 동일 ID 중복 카운트 방지
                self._confirm[tid] = self._confirm.get(tid, 0) + 1
                seen_tids.add(tid)

            ema_key = (tid, labels[i])                      # 레이블별 EMA 를 독립적으로 관리한다.
            box = boxes[i]
            # 직전 프레임에 빠졌다가 복귀한 트랙은 오래된 좌표와 섞지 않고 실제 좌표로 스냅한다.
            if ema_key in self._ema and self._miss.get(ema_key, 0) == 0:
                a   = config.EMA_ALPHA
                box = tuple(int(a * c + (1 - a) * p) for c, p in zip(box, self._ema[ema_key]))
            self._ema[ema_key]       = box
            self._miss[ema_key]      = 0                    # 검출됨 → coasting 카운터 초기화
            self._last_meta[ema_key] = (confs[i], cls_ids[i])

            if self._confirm[tid] >= config.MIN_CONFIRM_FRAMES:
                filtered.append((box, labels[i], confs[i], tid, cls_ids[i]))

        # 이번 프레임에 빠진 트랙 처리 : 확정 트랙은 마지막 박스를 잠시 유지(coasting)한다.
        for ema_key in set(self._ema) - current_ema_keys:
            tid, label = ema_key
            if self._confirm.get(tid, 0) < config.MIN_CONFIRM_FRAMES:
                self._ema.pop(ema_key, None)                # 미확정 트랙은 즉시 제거
                self._miss.pop(ema_key, None)
                self._last_meta.pop(ema_key, None)
                continue

            self._miss[ema_key] = self._miss.get(ema_key, 0) + 1
            if self._miss[ema_key] <= config.COAST_MAX_FRAMES:
                conf, cid = self._last_meta.get(ema_key, (0.0, 0))
                filtered.append((self._ema[ema_key], label, conf, tid, cid))
            else:                                           # 유지 한계 초과 → 완전히 제거
                self._ema.pop(ema_key, None)
                self._miss.pop(ema_key, None)
                self._last_meta.pop(ema_key, None)

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
                tracker="bytetrack.yaml", imgsz=config.INFER_IMGSZ, half=True
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

                # 헬멧 클래스에만 별도 신뢰도 임계값을 적용한다.
                HELMET_LABELS = {"helmet_O", "helmet_X"}
                keep   = [
                    i for i, (l, c) in enumerate(zip(labels, confs))
                    if l not in HELMET_LABELS or c >= config.HELMET_CONF
                ]
                boxes    = [boxes[i]    for i in keep]
                labels   = [labels[i]   for i in keep]
                confs    = [confs[i]    for i in keep]
                ids      = [ids[i]      for i in keep]
                cls_ids  = [cls_ids[i]  for i in keep]

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
