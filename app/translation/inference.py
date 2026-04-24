import os
import torch

MODEL_DIR = os.environ.get(
    "TRANSLATION_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
)


class Translator:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print("[translation] Loading EnViT5...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.model.cuda()
        print("[translation] Ready.")

    def en_to_vi(self, text: str) -> str:
        inputs = self.tokenizer(f"en: {text}", return_tensors="pt", padding=True).input_ids.cuda()
        with torch.no_grad():
            output = self.model.generate(inputs, max_length=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
