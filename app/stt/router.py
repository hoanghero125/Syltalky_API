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

SAMPLE_RATE = 16000


@router.post("/stt")
async def stt_file(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file (WAV/FLAC/MP3) to Vietnamese text."""
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
    Real-time Vietnamese speech transcription via VAD + offline ASR.

    Client sends: raw PCM float32 audio chunks (16kHz mono)
    Server sends: transcribed Vietnamese text per detected speech segment
    """
    await websocket.accept()

    loop = asyncio.get_event_loop()
    vad = transcriber.create_vad()
    window_size = transcriber.vad_window_size
    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            # Feed to VAD in fixed window_size chunks
            while len(buffer) >= window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            # Decode any completed speech segments
            while not vad.empty():
                segment = np.array(vad.front.samples, dtype=np.float32)
                vad.pop()

                text = await loop.run_in_executor(
                    executor, transcriber.transcribe_segment, segment
                )
                if text:
                    await websocket.send_text(text)

    except WebSocketDisconnect:
        # Flush remaining audio and send last segment if any
        vad.flush()
        while not vad.empty():
            segment = np.array(vad.front.samples, dtype=np.float32)
            vad.pop()
            text = await loop.run_in_executor(
                executor, transcriber.transcribe_segment, segment
            )
            if text:
                try:
                    await websocket.send_text(text)
                except Exception:
                    pass
