from fastapi import FastAPI, Request
from datetime import datetime

app = FastAPI()

# 수신된 위반 목록을 메모리에 저장한다.
received: list = []


# 함수 이름 : receive_violation()
# 기능      : kickboard_server.py 에서 전송한 위반 정보를 수신하여
#             터미널에 출력하고 received 목록에 저장한다.
# 파라미터  : Request request -> HTTP 요청 객체
# 반환값    : {"status": "ok", "received_at": str}
@app.post("/api/violations")
async def receive_violation(request: Request):
    body = await request.json()
    body["received_at"] = datetime.now().strftime("%H:%M:%S")
    received.append(body)

    print(f"\n[수신 #{len(received)}] {body['received_at']}")
    print(f"  유형      : {body.get('type')}")
    print(f"  카메라    : {body.get('camera')}")
    print(f"  신뢰도    : {body.get('confidence')}%")
    print(f"  누적 수신 : {len(received)}건")

    return {"status": "ok", "received_at": body["received_at"]}


# 함수 이름 : get_received()
# 기능      : 지금까지 수신된 위반 목록 전체를 반환한다.
# 반환값    : {"total": int, "violations": list}
@app.get("/api/violations")
async def get_received():
    return {"total": len(received), "violations": received}


if __name__ == "__main__":
    import uvicorn
    print("[더미 백엔드 시작] http://localhost:8080/api/violations")
    uvicorn.run(app, host="0.0.0.0", port=8080)