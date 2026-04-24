"""
Run once on the server to download the OpenASL checkpoint and MT5 weights.

    python download_model.py
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).parent / "app"

# Redirect HuggingFace cache to a local writable directory
os.environ["HF_HOME"] = str(Path(__file__).parent / ".hf_cache")


def download():
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer, MT5ForConditionalGeneration

    # 1. OpenASL checkpoint (best English SLT)
    checkpoint_dir = APP_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("Downloading openasl_pose_only_slt.pth...")
    hf_hub_download(
        repo_id="ZechengLi19/Uni-Sign",
        filename="openasl_pose_only_slt.pth",
        local_dir=str(checkpoint_dir),
    )
    print(f"Saved to {checkpoint_dir / 'openasl_pose_only_slt.pth'}")

    # 2. MT5-base
    mt5_dir = APP_DIR / "pretrained_weight" / "mt5-base"
    mt5_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading mt5-base...")
    T5Tokenizer.from_pretrained("google/mt5-base", legacy=False).save_pretrained(str(mt5_dir))
    MT5ForConditionalGeneration.from_pretrained("google/mt5-base").save_pretrained(str(mt5_dir))
    print(f"Saved to {mt5_dir}")

    print("\nDone.")


if __name__ == "__main__":
    download()
