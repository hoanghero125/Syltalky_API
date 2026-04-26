"""
Download all model weights for Syltalky API.

    python download_model.py
"""

import json
import os
import urllib.request
from pathlib import Path


def _patch_tokenizer_class(directory: Path, fast_class: str):
    cfg_path = directory / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") != fast_class:
        cfg["tokenizer_class"] = fast_class
        cfg_path.write_text(json.dumps(cfg, indent=2))

APP_DIR = Path(__file__).parent / "app"

os.environ["HF_HOME"] = str(Path(__file__).parent / "app" / "stt" / ".hf_cache")


def download():
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
    import sentencepiece as spm

    # ── 0. RTMPose ONNX models (rtmlib) ─────────────────────────────────────
    rtmlib_cache = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    yolox   = rtmlib_cache / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    rtmpose = rtmlib_cache / "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx"
    if yolox.exists() and rtmpose.exists():
        print("[0/6] RTMPose models already exist, skipping.")
    else:
        print("[0/6] Downloading RTMPose models (via rtmlib)...")
        import sys
        sys.path.insert(0, str(APP_DIR / "sign" / "rtmlib-main"))
        from rtmlib import Wholebody
        Wholebody(to_openpose=False, mode="lightweight", backend="onnxruntime", device="cpu")
        print(f"      → {rtmlib_cache}")

    # ── 1. Sign language: Uni-Sign checkpoint ────────────────────────────────
    sign_ckpt = APP_DIR / "sign" / "checkpoints"
    ckpt_file = sign_ckpt / "openasl_pose_only_slt.pth"
    if ckpt_file.exists():
        print("[1/6] Uni-Sign checkpoint already exists, skipping.")
    else:
        sign_ckpt.mkdir(parents=True, exist_ok=True)
        print("[1/6] Downloading Uni-Sign checkpoint...")
        hf_hub_download(
            repo_id="ZechengLi19/Uni-Sign",
            filename="openasl_pose_only_slt.pth",
            local_dir=str(sign_ckpt),
        )
        print(f"      → {ckpt_file}")

    # ── 2. Sign language: MT5-base ───────────────────────────────────────────
    mt5_dir = APP_DIR / "sign" / "pretrained_weight" / "mt5-base"
    if (mt5_dir / "config.json").exists():
        print("[2/6] MT5-base already exists, skipping.")
    else:
        mt5_dir.mkdir(parents=True, exist_ok=True)
        print("[2/6] Downloading MT5-base...")
        snapshot_download(repo_id="google/mt5-base", local_dir=str(mt5_dir), local_dir_use_symlinks=False)
        _patch_tokenizer_class(mt5_dir, "MT5TokenizerFast")
        print(f"      → {mt5_dir}")

    # ── 3. STT: Zipformer RNNT ───────────────────────────────────────────────
    stt_dir = APP_DIR / "stt" / "model"
    if (stt_dir / "silero_vad.onnx").exists():
        print("[3/6] STT model already exists, skipping.")
    else:
        stt_dir.mkdir(parents=True, exist_ok=True)
        print("[3/6] Downloading Zipformer STT model...")
        for f in [
            "encoder-epoch-20-avg-10.int8.onnx",
            "decoder-epoch-20-avg-10.int8.onnx",
            "joiner-epoch-20-avg-10.int8.onnx",
            "bpe.model",
        ]:
            hf_hub_download(repo_id="hynt/Zipformer-30M-RNNT-6000h", filename=f, local_dir=str(stt_dir))
            print(f"      {f}")

        sp = spm.SentencePieceProcessor()
        sp.Load(str(stt_dir / "bpe.model"))
        with open(stt_dir / "tokens.txt", "w") as f:
            for i in range(sp.GetPieceSize()):
                f.write(f"{sp.IdToPiece(i)} {i}\n")
        print("      tokens.txt (generated)")

        urllib.request.urlretrieve(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
            str(stt_dir / "silero_vad.onnx"),
        )
        print(f"      silero_vad.onnx")
        print(f"      → {stt_dir}")

    # ── 4. STT post-processing: punctuation + NER ────────────────────────────
    hf_cache     = Path(__file__).parent / "app" / "stt" / ".hf_cache" / "hub"
    punct_dir    = hf_cache / "models--oliverguhr--fullstop-punctuation-multilang-large"
    ner_dir      = hf_cache / "models--NlpHUST--ner-vietnamese-electra-base"
    punct_cached = punct_dir.exists() and any(punct_dir.rglob("*.safetensors"))
    ner_cached   = ner_dir.exists()   and any(ner_dir.rglob("*.safetensors"))
    if punct_cached and ner_cached:
        print("[4/6] STT post-processing models already exist, skipping.")
    else:
        print("[4/6] Downloading STT post-processing models...")
        from transformers import pipeline
        if not punct_cached:
            pipeline("ner", model="oliverguhr/fullstop-punctuation-multilang-large", device=-1)
            print("      oliverguhr/fullstop-punctuation-multilang-large")
        if not ner_cached:
            pipeline("ner", model="NlpHUST/ner-vietnamese-electra-base", aggregation_strategy="simple", device=-1)
            print("      NlpHUST/ner-vietnamese-electra-base")
        print(f"      → {hf_cache}")

    # ── 5. Translation: EnViT5 ───────────────────────────────────────────────
    trans_dir = APP_DIR / "translation" / "model"
    if (trans_dir / "config.json").exists():
        print("[5/6] EnViT5 translation model already exists, skipping.")
    else:
        trans_dir.mkdir(parents=True, exist_ok=True)
        print("[5/6] Downloading EnViT5 translation model...")
        snapshot_download(repo_id="VietAI/envit5-translation", local_dir=str(trans_dir), local_dir_use_symlinks=False)
        _patch_tokenizer_class(trans_dir, "T5TokenizerFast")
        print(f"      → {trans_dir}")

    # ── 6. TTS: OmniVoice Vietnamese ─────────────────────────────────────────
    tts_dir = APP_DIR / "tts" / "checkpoints" / "splendor1811" / "omnivoice-vietnamese"
    audio_tok_dir = tts_dir / "audio_tokenizer"
    if (tts_dir / "model.safetensors").exists() and audio_tok_dir.is_dir():
        print("[6/6] OmniVoice TTS model already exists, skipping.")
    else:
        tts_dir.mkdir(parents=True, exist_ok=True)
        print("[6/6] Downloading OmniVoice Vietnamese TTS model...")
        snapshot_download(
            repo_id="splendor1811/omnivoice-vietnamese",
            local_dir=str(tts_dir),
            local_dir_use_symlinks=False,
        )
        print(f"      → {tts_dir}")
        if not audio_tok_dir.is_dir():
            print("[6/6] Downloading HiggsAudio tokenizer (OmniVoice dependency)...")
            audio_tok_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="eustlb/higgs-audio-v2-tokenizer",
                local_dir=str(audio_tok_dir),
                local_dir_use_symlinks=False,
            )
            print(f"      → {audio_tok_dir}")

    print("\nAll models ready.")


if __name__ == "__main__":
    download()
