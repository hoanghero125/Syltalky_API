# Syltalky API

AI services for Syltalky.

---

## Services

| Endpoint | Method | Model | Description |
|---|---|---|---|
| `/sign` | POST | Uni-Sign + EnViT5 | ASL video → Vietnamese text |
| `/ws/stt` | WebSocket | Zipformer-RNNT + Silero VAD | Streaming speech → Vietnamese text |
| `/stt` | POST | Zipformer-RNNT + Silero VAD | Audio file → Vietnamese text |
| `/tts/voice` | POST | OmniVoice + HiggsAudio | Register a cloned voice, get `voice_id` |
| `/tts/synthesize` | POST | OmniVoice | Synthesize speech with a cloned voice |
| `/tts/design` | POST | OmniVoice | Synthesize speech with a designed voice |

---

## Requirements

- Python 3.11
- CUDA 12.6–12.8 capable GPU (8 GB+ VRAM recommended)
- conda

---

## Setup

```bash
# 1. Create conda environment
conda create -n syltalky-api python=3.11 -y
conda activate syltalky-api

# 2. Install dependencies (includes local rtmlib and omnivoice packages)
pip install -r requirements.txt

# 3. Start the server (downloads missing models automatically on first run)
python main.py
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

---

## API

### `GET /health`

```json
{ "status": "ok" }
```

---

### `POST /sign` — Sign Language → Vietnamese

Upload a recorded ASL video and receive Vietnamese text.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `video` | file | MP4, WebM, AVI, or MOV |

**Response:**
```json
{ "text": "Xin chào" }
```

**Notes:**
- Minimum 32 frames (~1s at 30fps)
- Pipeline: RTMPose keypoint extraction → Uni-Sign (ASL → EN) → EnViT5 (EN → VI)

---

### `WS /ws/stt` — Speech → Text (streaming)

Real-time Vietnamese speech transcription. Connect via WebSocket and stream raw audio.

**Client sends:** raw `float32` PCM chunks — 16 kHz, mono

**Server sends:** Vietnamese text string per detected speech segment (VAD-segmented)

---

### `POST /stt` — Speech → Text (file)

Transcribe an audio file to Vietnamese text.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `audio` | file | WAV, FLAC, or MP3 |

**Response:**
```json
{ "text": "Xin chào, tôi tên là Hoàng." }
```

**Post-processing pipeline:**

| Stage | Example |
|---|---|
| Raw ASR | `xin chào tôi tên là hoàng` |
| Punctuation | `xin chào, tôi tên là hoàng.` |
| NER capitalization | `xin chào, tôi tên là Hoàng.` |
| Sentence capitalization | `Xin chào, tôi tên là Hoàng.` |

---

### `POST /tts/voice` — Register a Cloned Voice

Upload a reference audio clip and its transcript to create a reusable voice. The audio is tokenized once and stored in memory. The returned `voice_id` is passed to `POST /tts/synthesize`.

**Intended pipeline (backend handles step 1):**
```
1. POST /stt  (ref audio)  →  transcript
2. POST /tts/voice  (ref audio + transcript)  →  voice_id
3. POST /tts/synthesize  (voice_id + text)  →  WAV   ← repeat as needed
```

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `ref_audio` | file | WAV recommended — 5–15s of clear speech, no background noise |
| `ref_text` | string | Exact transcript of what is spoken in `ref_audio` |

**Response:**
```json
{
  "voice_id": "3f2a1b4c-...",
  "transcript": "Xin chào, tôi tên là Hoàng."
}
```

**Notes:**
- `voice_id` persists in memory until the server restarts
- The backend is responsible for storing `voice_id` per user and re-registering if the server restarts
- Reference audio longer than 20s will produce a warning and may reduce quality — trim to 15s or under for best results

---

### `POST /tts/synthesize` — Synthesize with Cloned Voice

Synthesize Vietnamese text using a voice previously registered with `POST /tts/voice`. The reference audio is never re-processed — this goes straight to the diffusion step.

**Request:** `application/json`

```json
{
  "voice_id": "3f2a1b4c-...",
  "text": "Hôm nay trời đẹp quá.",
  "num_step": 32,
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `voice_id` | string | required | ID returned by `POST /tts/voice` |
| `text` | string | required | Vietnamese text to synthesize |
| `num_step` | int | 32 | Diffusion steps (1–128). Higher = better quality, slower |
| `speed` | float | 1.0 | Playback speed (0.5–2.0) |

**Response:** `audio/wav`

---

### `POST /tts/design` — Synthesize with Designed Voice

Synthesize Vietnamese text using a voice described by comma-separated style tags. No reference audio needed.

**Request:** `application/json`

```json
{
  "text": "Xin chào, rất vui được gặp bạn.",
  "instruct": "female, young adult, high pitch",
  "num_step": 32,
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Vietnamese text to synthesize |
| `instruct` | string | required | Comma-separated style tags (see table below) |
| `num_step` | int | 32 | Diffusion steps (1–128) |
| `speed` | float | 1.0 | Playback speed (0.5–2.0) |

**Valid tags:**

| Category | Tags |
|---|---|
| Gender | `female` · `male` |
| Age | `child` · `teenager` · `young adult` · `middle-aged` · `elderly` |
| Pitch | `very low pitch` · `low pitch` · `moderate pitch` · `high pitch` · `very high pitch` |
| Style | `whisper` |
| Accent | `american accent` · `australian accent` · `british accent` · `canadian accent` · `chinese accent` · `indian accent` · `japanese accent` · `korean accent` · `portuguese accent` · `russian accent` |

Combine tags freely, one per category: `"female, young adult, high pitch, british accent"`

**Response:** `audio/wav`

---

## Project structure

```
Syltalky_API/
├── main.py                 ← sets HF_HOME before any import, starts uvicorn
├── download_model.py       ← downloads all models (called automatically by main.py)
├── requirements.txt
├── demo.html               ← browser demo (sign + STT + TTS)
└── app/
    ├── api.py              ← FastAPI app, mounts all routers
    ├── sign/               ← ASL → Vietnamese (Uni-Sign + RTMPose + EnViT5)
    │   ├── router.py       ← POST /sign
    │   ├── inference.py
    │   ├── rtmlib-main/    ← bundled rtmlib (pip install -e)
    │   ├── checkpoints/    ← openasl_pose_only_slt.pth (gitignored)
    │   └── pretrained_weight/ ← mt5-base/ (gitignored)
    ├── stt/                ← Vietnamese speech → text
    │   ├── router.py       ← POST /stt, WS /ws/stt
    │   ├── inference.py
    │   ├── model/          ← Zipformer ONNX + bpe.model + silero_vad.onnx (gitignored)
    │   └── .hf_cache/      ← HF cache for punct + NER models (gitignored)
    ├── translation/        ← EN → VI (EnViT5, used internally by sign)
    │   ├── inference.py
    │   └── model/          ← EnViT5 weights (gitignored)
    └── tts/                ← Vietnamese text → speech (OmniVoice)
        ├── router.py       ← POST /tts/voice, /tts/synthesize, /tts/design
        ├── inference.py
        ├── omnivoice/      ← bundled OmniVoice source (pip install -e)
        ├── speakers/       ← preset speaker ref audio (reserved)
        └── checkpoints/    ← omnivoice-vietnamese weights (gitignored)
```

---

## Credits

- [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) — Li et al., ICLR 2025
- [OmniVoice](https://github.com/k2-fsa/OmniVoice) — k2-fsa / splendor1811
- [Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) — hynt, VLSP 2025
- [EnViT5](https://huggingface.co/VietAI/envit5-translation) — VietAI
- [RTMPose / rtmlib](https://github.com/Tau-J/rtmlib) — Tau-J
- [fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) — Oliver Guhr
- [ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base) — NlpHUST
- [Silero VAD](https://github.com/snakers4/silero-vad) — snakers4 (via sherpa-onnx)
- [HiggsAudio v2 Tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) — eustlb
