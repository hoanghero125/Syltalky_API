import io
import os
import uuid
import torch
import torchaudio
from pathlib import Path

HERE     = Path(__file__).resolve().parent
SPEAKERS = HERE / "speakers"

MODEL_ID = os.environ.get("TTS_MODEL_ID", "splendor1811/omnivoice-vietnamese")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


class TTSEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._voices = {}
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load()

    def _load(self):
        from omnivoice import OmniVoice
        # Prefer local checkpoint if downloaded
        local = HERE / "checkpoints" / MODEL_ID
        path  = str(local) if (local / "model.safetensors").exists() else MODEL_ID
        print(f"[tts] Loading OmniVoice from {path}...")
        self._model = OmniVoice.from_pretrained(path, device_map=DEVICE, dtype=torch.float16)
        print("[tts] Ready.")

    def _to_wav_bytes(self, audio: torch.Tensor) -> bytes:
        audio = audio.detach().cpu().to(torch.float32)
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            audio = audio.squeeze()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
        audio = audio - audio.mean()
        peak = torch.abs(audio).max()
        if peak > 0:
            audio = audio * (0.95 / peak)
        buf = io.BytesIO()
        torchaudio.save(buf, audio, self._model.sampling_rate, format="wav",
                        encoding="PCM_S", bits_per_sample=16)
        buf.seek(0)
        return buf.read()

    def create_voice(self, ref_audio_path: str, ref_text: str = None) -> tuple:
        """Tokenize reference audio once and store as a reusable voice prompt."""
        prompt = self._model.create_voice_clone_prompt(
            ref_audio_path, ref_text=ref_text or None
        )
        voice_id = str(uuid.uuid4())
        self._voices[voice_id] = prompt
        return voice_id, prompt.ref_text

    def synthesize_with_voice(self, voice_id: str, text: str,
                               num_step: int = 32, speed: float = 1.0) -> bytes:
        prompt = self._voices.get(voice_id)
        if not prompt:
            raise ValueError(f"Voice '{voice_id}' not found.")
        audios = self._model.generate(
            text=text, voice_clone_prompt=prompt, num_step=num_step, speed=speed
        )
        return self._to_wav_bytes(audios[0])

    def synthesize(self, text: str, ref_audio: str = None, ref_text: str = None,
                   instruct: str = None, num_step: int = 32, speed: float = 1.0) -> bytes:
        kwargs = {"text": text, "num_step": num_step, "speed": speed}
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct
        audios = self._model.generate(**kwargs)
        return self._to_wav_bytes(audios[0])
