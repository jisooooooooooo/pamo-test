import cv2
import asyncio
import base64
import json
from fastapi import FastAPI, WebSocket
from motion_detector import MotionDetector
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

MOTION_MAP = {
    "손을 위로 들기": {"id": "raise_hand", "label": "손을 위로 들기"},
    "손을 좌우로 흔들기": {"id": "wave_hand", "label": "손을 좌우로 흔들기"},
    "손을 앞으로 뻗기": {"id": "push_hand", "label": "손을 앞으로 뻗기"},
    "고개 좌우로 흔들기": {"id": "shake_head", "label": "고개 좌우로 흔들기"},
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera = cv2.VideoCapture(0)
    detector = MotionDetector()
    prev_motions = set()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            # 감지 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_motions = detector.detect(frame_rgb)
            new_motions = current_motions - prev_motions
            if new_motions:
                motion_list = []
                for motion in new_motions:
                    if motion in MOTION_MAP:
                        motion_list.append(MOTION_MAP[motion])

                motion_message = {
                    "type": "motion",
                    "motions": motion_list
                }
                try:
                    await websocket.send_text(json.dumps(motion_message, ensure_ascii=False))
                except WebSocketDisconnect:
                    print("클라이언트가 연결을 끊음 (motion)")
                    break

                prev_motions = current_motions.copy()

            frame = cv2.flip(frame, 1)
            _, buffer = cv2.imencode('.jpg', frame)

            try:
                await websocket.send_bytes(buffer.tobytes())
            except WebSocketDisconnect:
                print("클라이언트가 연결을 끊음 (frame)")
                break

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("클라이언트 연결 해제됨")
    finally:
        camera.release()
