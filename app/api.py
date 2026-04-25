import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.sign.router import router as sign_router
from app.stt.router import router as stt_router
from app.tts.router import router as tts_router

app = FastAPI(title="Syltalky API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sign_router, tags=["Sign Language"])
app.include_router(stt_router, tags=["STT"])
app.include_router(tts_router, tags=["TTS"])


@app.get("/health")
async def health():
    return {"status": "ok"}
