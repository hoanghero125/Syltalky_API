# Syltalky API

AI service for sign language translation. Accepts a video of someone signing and returns an English translation.

Built on top of [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) (ICLR 2025), using the OpenASL checkpoint (pose-only, ~97K clips).

---

## How it works

1. Extracts 133 body/hand/face keypoints per frame using RTMPose (ONNX)
2. Encodes the keypoint sequence through a Spatial-Temporal GCN
3. Decodes to English text via MT5

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

# 2. Install rtmlib
pip install -e app/rtmlib-main

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights (~1.5 GB)
python download_model.py

# 5. Start the server
python main.py
```

Server runs at `http://localhost:8000`.

---

## API

### `GET /health`
Returns server status.

```json
{ "status": "ok" }
```

### `POST /translate`
Accepts a video file (MP4 or WebM) and returns the English translation.

**Request:** `multipart/form-data` with field `video`

**Response:**
```json
{ "translation": "I want to go to the store" }
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/translate \
  -F "video=@signing.mp4"
```

---

## Project structure

```
Syltalky_API/
├── main.py               # Entry point
├── download_model.py     # One-time model download
├── requirements.txt
└── app/
    ├── api.py            # FastAPI routes
    ├── inference.py      # Translation pipeline
    ├── models.py         # Uni-Sign model architecture
    ├── datasets.py       # Data loading utilities
    ├── config.py         # Path configuration
    ├── utils.py          # Utilities
    ├── deformable_attention_2d.py
    ├── stgcn_layers/     # Spatial-Temporal GCN
    └── rtmlib-main/      # RTMPose keypoint extractor
```

---

## Credits

- [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) — Li et al., ICLR 2025
- [RTMPose](https://github.com/open-mmlab/mmpose) — MMPose team
