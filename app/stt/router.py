import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from app.stt.inference import SpeechTranscriber

router = APIRouter()
transcriber = SpeechTranscriber()
executor = ThreadPoolExecutor(max_workers=2)


@router.post("/stt")
async def stt_file(audio: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file (WAV/FLAC/MP3) to Vietnamese text.
    """
    try:
        data = await audio.read()
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, transcriber.transcribe, samples, sr)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/stt")
async def stt_stream(websocket: WebSocket):
    """
    Real-time Vietnamese speech transcription from microphone.

    Client sends: raw PCM float32 audio chunks (16kHz mono)
    Server sends: transcribed Vietnamese text string
    """
    await websocket.accept()

    loop = asyncio.get_event_loop()
    stream = transcriber.recognizer.create_stream()
    SAMPLE_RATE = 16000

    try:
        while True:
            data = await websocket.receive_bytes()

            # data is raw float32 PCM bytes
            chunk = np.frombuffer(data, dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, chunk)

            while transcriber.recognizer.is_ready(stream):
                transcriber.recognizer.decode_stream(stream)

            result = transcriber.recognizer.get_result(stream).text.strip()
            if result:
                await websocket.send_text(result)

    except WebSocketDisconnect:
        pass
