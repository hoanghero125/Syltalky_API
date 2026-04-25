import os
import uvicorn

# Must be set before any huggingface_hub/transformers import resolves cache paths
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(__file__), "app", "stt", ".hf_cache"),
)

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
