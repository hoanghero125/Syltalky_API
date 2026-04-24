import os
import tempfile
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.inference import SignTranslator

app = FastAPI(title="Syltalky Sign Language Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = SignTranslator()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate(video: UploadFile = File(...)):
    """
    Accept a video file (WebM/MP4) of someone signing and return the English translation.
    """
    suffix = Path(video.filename).suffix if video.filename else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        text = translator.translate(tmp_path)
        return {"translation": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


