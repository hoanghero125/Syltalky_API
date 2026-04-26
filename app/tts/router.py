import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.tts.inference import TTSEngine, SPEAKERS

router   = APIRouter()
engine   = TTSEngine()
executor = ThreadPoolExecutor(max_workers=1)

UPLOAD_DIR = Path("uploads/tts")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── schemas ──────────────────────────────────────────────────────────────────

_VALID_TAGS = """
**Valid tags** (comma-separated, combine freely):

| Category | Tags |
|---|---|
| Gender | `female`, `male` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| Style | `whisper` |
| Accent | `american accent`, `australian accent`, `british accent`, `canadian accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent` |

**Examples:** `"female, young adult, high pitch"` · `"male, elderly, low pitch, british accent"` · `"female, whisper"`
"""

class DesignRequest(BaseModel):
    text:     str   = Field(..., description="Vietnamese text to synthesize")
    instruct: str   = Field(..., description=_VALID_TAGS, example="female, young adult, high pitch")
    num_step: int   = Field(32,  ge=1, le=128)
    speed:    float = Field(1.0, ge=0.5, le=2.0)

class SynthesizeRequest(BaseModel):
    voice_id: str   = Field(..., description="Voice ID returned by POST /tts/voice")
    text:     str   = Field(..., description="Vietnamese text to synthesize")
    num_step: int   = Field(32,  ge=1, le=128)
    speed:    float = Field(1.0, ge=0.5, le=2.0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _wav_response(wav_bytes: bytes) -> StreamingResponse:
    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts.wav"},
    )

async def _run(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/tts/voice", summary="Create a reusable voice from reference audio")
async def create_voice(
    ref_audio: UploadFile = File(..., description="WAV file, 5–15s of clear speech"),
    ref_text:  str        = Form(..., description="Transcript of the reference audio"),
):
    """Tokenize a reference audio clip once and return a `voice_id`.

    Pass `voice_id` to `POST /tts/synthesize` as many times as needed —
    the audio is never re-processed. Voices are kept in memory until server restart."""
    data = await ref_audio.read()
    tmp  = UPLOAD_DIR / f"{uuid.uuid4()}_{ref_audio.filename}"
    tmp.write_bytes(data)
    try:
        voice_id, transcript = await _run(engine.create_voice, str(tmp), ref_text)
    finally:
        tmp.unlink(missing_ok=True)
    return {"voice_id": voice_id, "transcript": transcript}


@router.post("/tts/synthesize", summary="Synthesize with a created voice")
async def tts_synthesize(request: SynthesizeRequest):
    """Synthesize Vietnamese text using a voice created with `POST /tts/voice`.

    Goes straight to diffusion — no audio re-processing."""
    try:
        wav = await _run(engine.synthesize_with_voice,
                         request.voice_id, request.text, request.num_step, request.speed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _wav_response(wav)


@router.post("/tts/design", summary="Synthesize with a voice described by instruction")
async def tts_design(request: DesignRequest):
    """Synthesize Vietnamese text using a voice described by comma-separated tags.
    See the `instruct` field description for all valid tags."""
    wav = await _run(engine.synthesize, request.text,
                     None, None,
                     request.instruct, request.num_step, request.speed)
    return _wav_response(wav)
