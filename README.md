# Syltalky API

AI service layer for Syltalky — a real-time communication app for deaf people in online calls.

---

## Concept

Syltalky bridges deaf and hearing people in a call. The deaf person signs via webcam; their signing is translated and spoken aloud to the hearing person. The hearing person speaks normally; their speech is transcribed as text for the deaf person to read.

```
┌─────────────────────────────────────────────────────────────┐
│                        DEAF PERSON                          │
│                                                             │
│  [Webcam]                              [Screen / subtitles] │
│     │                                          ▲            │
│     ▼                                          │            │
│  WS /ws/translate                       WS /ws/stt         │
│  ASL → English                    Vietnamese speech → text  │
│     │                                          │            │
│     ▼                                          │            │
│  EN → VI translation                    [Call audio in]     │
│     │                                                       │
│     ▼                                                       │
│  WS /ws/tts                                                 │
│  Vietnamese text → speech                                   │
│     │                                                       │
│     ▼                                                       │
│  [Virtual mic → call output]                                │
│                          HEARING PERSON hears Vietnamese    │
└─────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Endpoint | Status | Model |
|---|---|---|---|
| Sign Language Translation | `WS /ws/translate` | ✅ Done | Uni-Sign (OpenASL) + EnViT5 |
| Speech-to-Text | `WS /ws/stt`, `POST /stt` | ✅ Done | Zipformer-30M-RNNT (6000h Vietnamese) + Silero VAD |
| Text-to-Speech | `WS /ws/tts` | 🔜 Planned | TBD |

---

## Requirements

- Python 3.9
- CUDA-capable GPU (4GB+ VRAM)
- conda

---

## Setup

```bash
# 1. Create conda environment
conda create -n syltalky python=3.9 -y
conda activate syltalky

# 2. Install rtmlib (keypoint extractor for sign translation)
pip install -e app/sign/rtmlib-main

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download all model weights
python download_model.py

# 5. Start the server
python main.py
```

Server runs at `http://localhost:8000`.

---

## API

### `GET /health`
```json
{ "status": "ok" }
```

### `WS /ws/translate` — Sign Language Translation
Real-time ASL → Vietnamese translation.

- **Client sends:** raw JPEG frame bytes (~30fps from webcam)
- **Server sends:** Vietnamese text (every ~64 frames / ~2 seconds)
- Internally: ASL → English (Uni-Sign) → Vietnamese (EnViT5)

### `WS /ws/stt` — Speech-to-Text
Real-time Vietnamese speech transcription via VAD-segmented offline ASR.

- **Client sends:** raw float32 PCM audio chunks (16kHz mono)
- **Server sends:** Vietnamese text per detected speech segment (~1–2s latency)
- Internally: Silero VAD detects speech boundaries → Zipformer-RNNT transcribes each segment → post-processing

### `POST /stt` — Speech-to-Text (file upload)
Transcribe an audio file (WAV/FLAC/MP3) to Vietnamese text.

- **Request:** `multipart/form-data` with field `audio`
- **Response:** `{ "text": "Xin chào, tôi tên là Hoàng." }`

The STT output goes through a three-stage post-processing pipeline:

| Stage | Input | Output |
|---|---|---|
| Punctuation restoration | `xin chào tôi tên là hoàng` | `xin chào, tôi tên là hoàng.` |
| NER capitalization | `xin chào, tôi tên là hoàng.` | `xin chào, tôi tên là Hoàng.` |
| Sentence capitalization | `xin chào, tôi tên là Hoàng.` | `Xin chào, tôi tên là Hoàng.` |

---

## Project structure

```
Syltalky_API/
├── main.py                 ← sets HF_HOME before any import
├── download_model.py
├── requirements.txt
├── demo.html
└── app/
    ├── api.py              ← root app, mounts all routers
    ├── sign/               ← ASL → English (Uni-Sign)
    │   ├── router.py
    │   ├── inference.py
    │   ├── models.py
    │   ├── datasets.py
    │   ├── config.py
    │   ├── utils.py
    │   ├── deformable_attention_2d.py
    │   ├── stgcn_layers/
    │   ├── rtmlib-main/
    │   ├── checkpoints/            ← openasl_pose_only_slt.pth
    │   └── pretrained_weight/      ← mt5-base/
    ├── stt/                ← Vietnamese speech → text (Zipformer + VAD)
    │   ├── router.py
    │   ├── inference.py
    │   ├── model/                  ← ONNX encoder/decoder/joiner + bpe.model + tokens.txt + silero_vad.onnx
    │   └── .hf_cache/              ← HF model cache (punctuation + NER models)
    ├── translation/        ← EN → VI (EnViT5, used internally by sign)
    │   └── inference.py
    └── tts/                ← Vietnamese text → speech (planned)
        └── router.py
```

---

## Credits

- [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) — Li et al., ICLR 2025
- [Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) — hynt, VLSP 2025
- [EnViT5](https://huggingface.co/VietAI/envit5-translation) — VietAI
- [RTMPose](https://github.com/open-mmlab/mmpose) — MMPose team
- [fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) — Oliver Guhr
- [ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base) — NlpHUST
- [Silero VAD](https://github.com/snakers4/silero-vad) — snakers4 (via sherpa-onnx)
