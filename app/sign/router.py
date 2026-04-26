import asyncio
import re
import os
from concurrent.futures import ThreadPoolExecutor

import cv2

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.sign.inference import SignTranslator
from app.translation.inference import Translator

router = APIRouter()
sign_translator = SignTranslator()
translator = Translator()
executor = ThreadPoolExecutor(max_workers=2)

_MIN_SIGN_FRAMES = 32
_SPEAKER_RE = re.compile(r'^[A-Z][A-Za-z\s\'.]+:\s*')
_BRACKET_RE = re.compile(r'\[.*?\]', re.DOTALL)


def _translate_pipeline(pose_data):
    english = sign_translator.translate(pose_data)
    english = _BRACKET_RE.sub('', english)
    english = _SPEAKER_RE.sub('', english).strip()
    if not english:
        return ''
    return translator.en_to_vi(english)


@router.post("/sign")
async def sign_translate_file(video: UploadFile = File(...)):
    """Translate an uploaded ASL video file (MP4, WebM, AVI, MOV) to Vietnamese."""
    import tempfile
    try:
        data = await video.read()
        ext = ".webm" if "webm" in (video.content_type or "") else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        pose_data = {"keypoints": [], "scores": []}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kpts, scores = sign_translator.extract_keypoints(frame)
            pose_data["keypoints"].append(kpts)
            pose_data["scores"].append(scores)
        cap.release()
        os.unlink(tmp_path)

        if len(pose_data["keypoints"]) < _MIN_SIGN_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Video too short (need at least {_MIN_SIGN_FRAMES} frames)"
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _translate_pipeline, pose_data)
        return {"text": result or "No translation produced"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
