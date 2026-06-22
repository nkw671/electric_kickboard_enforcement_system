import os
import uvicorn
import torch
from threading import Thread
from ultralytics import YOLO

import config
from zone import ZoneDrawer, ZoneRenderer
from violation import DecideViolation, build_strategies
from detection import DetectionLoop
from api import ConnectAPI


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True          # 고정 입력 크기에 CUDA 자동 최적화
    # YOLO 모델을 로드한다.
    model = YOLO(config.MODEL_PATH).to("cuda")
    model.fuse()                                   # Conv+BN 레이어 융합으로 추론 속도 향상
    print(model.device)
    print("YOLO model loaded successfully!")

    # Zone 데이터 관리 객체를 생성하고 저장된 Zone 을 불러온다.
    drawer = ZoneDrawer()
    if os.path.exists(config.ZONE_FILE):
        drawer.load(config.ZONE_FILE)

    renderer = ZoneRenderer(drawer)

    api = ConnectAPI(drawer=drawer, alert_history=config.alert_history)

    decider = DecideViolation(
        strategies   = build_strategies(drawer),
        on_violation = api.send_violation,
    )

    loop = DetectionLoop(model=model, renderer=renderer, decider=decider)
    Thread(target=loop.run, daemon=True).start()

    uvicorn.run(api.app, host="0.0.0.0", port=8000)
