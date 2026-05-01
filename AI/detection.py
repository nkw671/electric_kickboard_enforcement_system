import cv2
from ultralytics.utils.plotting import Annotator, colors as yolo_colors
import config


# 클래스명        : DetectionLoop
# 기능           : 영상 캡처 및 YOLO 추론 루프 전담 (SRP 분리)
# 내장 함수 목록  : __init__() - 모델, 렌더러, 판정기 초기화
#                  run()      - 영상 캡처, YOLO 추론, 프레임 인코딩 루프 실행
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

    # 함수 이름 : run()
    # 기능      : 영상을 프레임 단위로 읽고 YOLO 로 추적하면서 decider.check() 를 호출한다.
    #             처리된 프레임을 JPEG 로 인코딩하여 config.latest_frame 에 저장한다.
    # 파라미터  : 없음
    # 반환값    : 없음
    def run(self):
        cap = cv2.VideoCapture(config.SOURCE)
        print("\n[감지 루프 시작]\n")

        while True:
            ret, frame = cap.read()
            if not ret:                         # 영상 끝에 도달하면 처음부터 재생한다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # YOLO 로 객체를 추적한다.
            results = self.model.track(
                frame, persist=True, conf=config.CONF, verbose=False,
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
            _, buf = cv2.imencode(".jpg", frame, config.ENCODE_PARAMS)
            config.latest_frame = buf.tobytes()

        cap.release()
