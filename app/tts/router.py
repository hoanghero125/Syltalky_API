from fastapi import APIRouter

router = APIRouter()


@router.post("/tts")
async def text_to_speech():
    # TODO
    pass
