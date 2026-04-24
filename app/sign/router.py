import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.sign.inference import SignTranslator
from app.translation.inference import Translator

router = APIRouter()
sign_translator = SignTranslator()
translator = Translator()
executor = ThreadPoolExecutor(max_workers=2)

CHUNK_FRAMES = 64
OVERLAP_FRAMES = 16


def _translate_pipeline(pose_data: dict) -> str:
    english = sign_translator.translate(pose_data)
    vietnamese = translator.en_to_vi(english)
    return vietnamese


@router.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """
    Real-time sign language translation from webcam frames.

    Client sends: raw JPEG frame bytes
    Server sends: Vietnamese text string
    """
    await websocket.accept()

    keypoints_buf = []
    scores_buf = []
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            kpts, scores = await loop.run_in_executor(
                executor, sign_translator.extract_keypoints, frame
            )
            keypoints_buf.append(kpts)
            scores_buf.append(scores)

            if len(keypoints_buf) >= CHUNK_FRAMES:
                pose_data = {
                    "keypoints": list(keypoints_buf),
                    "scores": list(scores_buf),
                }

                result = await loop.run_in_executor(
                    executor, _translate_pipeline, pose_data
                )

                await websocket.send_text(result)

                keypoints_buf = keypoints_buf[CHUNK_FRAMES - OVERLAP_FRAMES:]
                scores_buf = scores_buf[CHUNK_FRAMES - OVERLAP_FRAMES:]

    except WebSocketDisconnect:
        pass
