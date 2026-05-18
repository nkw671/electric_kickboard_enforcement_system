import queue
import threading
import time
import cv2
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
import config


# 클래스명        : DetectionLoop
# 기능           : 영상 캡처 및 YOLO 추론 루프 전담 (SRP 분리)
# 내장 함수 목록  : __init__()   - 모델, 렌더러, 판정기 초기화
#                  _capture()   - 별도 스레드에서 프레임 캡처 및 큐 공급
#                  run()        - 큐에서 프레임을 받아 YOLO 추론 및 인코딩 루프 실행
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
                tracker="bytetrack.yaml", vid_stride=2
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
            _, buf = cv2.imencode(".jpg", frame, config.ENCODE_PARAMS)
            config.latest_frame = buf.tobytes()
