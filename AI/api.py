import asyncio
import httpx
from datetime import datetime
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import config


# 클래스명        : ConnectAPI
# 기능           : API 통신 전담 — FastAPI 엔드포인트 등록 및 백엔드 위반 전송
# 내장 함수 목록  : __init__()          - FastAPI 앱, CORS, 엔드포인트 등록
#                  send_violation()    - 위반 정보를 별도 스레드로 백엔드에 전송 (논블로킹)
#                  _post_violation()   - 백엔드 POST /api/violations 실제 전송 (스레드에서 호출)
#                  video_stream()      - latest_frame 을 MJPEG 형식으로 실시간 스트리밍
#                  get_zones()         - 현재 설정된 Zone 목록 반환
#                  set_zones()         - React 캔버스에서 그린 Zone 목록을 받아 저장
#                  delete_zones()      - 모든 Zone 초기화 후 파일 갱신
#                  get_alerts()        - 누적된 위반 알림 목록 반환
class ConnectAPI:

    # 함수 이름 : __init__()
    # 기능      : FastAPI 앱을 생성하고 CORS 설정 및 엔드포인트를 등록한다.
    # 파라미터  : ZoneDrawer drawer        -> 감지 루프와 공유하는 ZoneDrawer 인스턴스
    #             list       alert_history -> 감지 루프와 공유하는 알림 목록
    # 반환값    : 없음
    def __init__(self, drawer, alert_history: list):
        self.drawer        = drawer
        self.alert_history = alert_history

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.get("/video/stream")(self.video_stream)
        self.app.get("/zones")(self.get_zones)
        self.app.post("/zones")(self.set_zones)
        self.app.delete("/zones")(self.delete_zones)
        self.app.get("/alerts")(self.get_alerts)

    # 함수 이름 : send_violation()
    # 기능      : 위반 정보를 alert_history 에 추가하고
    #             백엔드 전송을 별도 스레드에서 실행하여 감지 루프를 블로킹하지 않는다.
    # 파라미터  : str   violation_type -> 위반 유형
    #             int   track_id       -> 객체 추적 ID
    #             float conf           -> YOLO 감지 신뢰도 (0.0 ~ 1.0)
    # 반환값    : 없음
    def send_violation(self, violation_type: str, track_id: int, conf: float):
        payload = {
            "type":       violation_type,
            "image_url":  "",   # TODO: 나중에 실제 image_url 로 교체
            "camera":     config.CAMERA_ID,
            "confidence": int(conf * 100),
        }

        alert = {
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "type":       violation_type,
            "track_id":   track_id,
            "confidence": payload["confidence"],
            "camera":     config.CAMERA_ID,
        }
        self.alert_history.append(alert)
        print(alert)

        # 전송을 별도 스레드에서 실행하여 감지 루프를 블로킹하지 않는다.
        Thread(target=self._post_violation, args=(payload,), daemon=True).start()

    # 함수 이름 : _post_violation()
    # 기능      : 백엔드로 위반 정보를 전송한다.
    #             send_violation() 에서 별도 스레드로 호출된다.
    # 파라미터  : dict payload -> 전송할 위반 정보 딕셔너리
    # 반환값    : 없음
    def _post_violation(self, payload: dict):
        try:
            with httpx.Client() as client:
                client.post(config.BACKEND_URL, json=payload, timeout=3.0)
        except Exception as e:
            print(f"[전송 실패] {payload.get('type')} | {e}")

    # 함수 이름 : video_stream()
    # 기능      : config.latest_frame 을 MJPEG 형식으로 실시간 스트리밍한다.
    # 파라미터  : 없음
    # 반환값    : StreamingResponse (multipart/x-mixed-replace)
    async def video_stream(self):
        async def generate():
            while True:
                if config.latest_frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + config.latest_frame +
                        b"\r\n"
                    )
                await asyncio.sleep(0.03)
        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # 함수 이름 : get_zones()
    # 기능      : 현재 설정된 Zone 목록을 반환한다.
    # 파라미터  : 없음
    # 반환값    : {"zones": [...]}
    async def get_zones(self):
        return {
            "zones": [
                {"name": z["name"], "pts": z["pts"], "color": list(z["color"])}
                for z in self.drawer.zones
            ]
        }

    # 함수 이름 : set_zones()
    # 기능      : React 캔버스에서 그린 Zone 목록을 받아 ZoneDrawer 에 저장한다.
    #             Zone 변환 로직은 ZoneDrawer.set_zones() 에서 처리한다. (SRP)
    # 파라미터  : body = {"zones": [{"name": str, "pts": [[x,y],...], "color": [B,G,R]}, ...]}
    # 반환값    : {"saved": Zone 개수}
    async def set_zones(self, body: dict):
        self.drawer.set_zones(body.get("zones", []))
        return {"saved": len(self.drawer.zones)}

    # 함수 이름 : delete_zones()
    # 기능      : 모든 Zone 을 초기화하고 파일을 갱신한다.
    # 파라미터  : 없음
    # 반환값    : {"cleared": true}
    async def delete_zones(self):
        self.drawer.zones = []
        self.drawer.save(config.ZONE_FILE)
        return {"cleared": True}

    # 함수 이름 : get_alerts()
    # 기능      : 누적된 위반 알림 목록을 반환한다.
    #             after 파라미터로 특정 인덱스 이후 알림만 조회할 수 있다.
    # 파라미터  : int after -> 이 인덱스 이후의 알림만 반환 (기본값 0)
    # 반환값    : {"alerts": [...], "total": int}
    async def get_alerts(self, after: int = 0):
        return {
            "alerts": self.alert_history[after:],
            "total":  len(self.alert_history),
        }
