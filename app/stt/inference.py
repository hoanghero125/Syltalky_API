import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get(
    "STT_MODEL_DIR",
    os.path.join(HERE, "model"),
)

_NER_ENTITY_GROUPS = {"PERSON", "LOCATION", "ORGANIZATION"}


class SpeechTranscriber:
    def __init__(self):
        import sherpa_onnx
        from deepmultilingualpunctuation import PunctuationModel
        from transformers import pipeline

        encoder = os.path.join(MODEL_DIR, "encoder-epoch-20-avg-10.int8.onnx")
        decoder = os.path.join(MODEL_DIR, "decoder-epoch-20-avg-10.int8.onnx")
        joiner  = os.path.join(MODEL_DIR, "joiner-epoch-20-avg-10.int8.onnx")
        tokens  = os.path.join(MODEL_DIR, "tokens.txt")
        bpe     = os.path.join(MODEL_DIR, "bpe.model")
        silero  = os.path.join(MODEL_DIR, "silero_vad.onnx")

        print("[stt] Loading Zipformer RNNT...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            modeling_unit="bpe",
            bpe_vocab=bpe,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
        )

        print("[stt] Loading VAD...")
        self._vad_config = sherpa_onnx.VadModelConfig()
        self._vad_config.silero_vad.model = silero
        self._vad_config.silero_vad.threshold = 0.5
        self._vad_config.silero_vad.min_silence_duration = 0.5
        self._vad_config.silero_vad.min_speech_duration = 0.25
        self._vad_config.sample_rate = 16000
        self.vad_window_size = self._vad_config.silero_vad.window_size

        print("[stt] Loading punctuation model...")
        self.punct_model = PunctuationModel()

        print("[stt] Loading NER model...")
        self.ner = pipeline(
            "ner",
            model="NlpHUST/ner-vietnamese-electra-base",
            aggregation_strategy="simple",
            device=0,
        )

        print("[stt] Ready.")

    def create_vad(self):
        import sherpa_onnx
        return sherpa_onnx.VoiceActivityDetector(
            self._vad_config, buffer_size_in_seconds=60
        )

    def _postprocess(self, text: str) -> str:
        text = text.lower()
        text = self.punct_model.restore_punctuation(text)

        entities = self.ner(text)
        chars = list(text)
        for ent in entities:
            if ent["entity_group"] not in _NER_ENTITY_GROUPS:
                continue
            capitalize_next = True
            for i in range(ent["start"], ent["end"]):
                if chars[i] == " ":
                    capitalize_next = True
                elif capitalize_next:
                    chars[i] = chars[i].upper()
                    capitalize_next = False
        text = "".join(chars)

        text = re.sub(
            r"(^|(?<=[.!?])\s+)(\w)",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        return text

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio array (for POST /stt)."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_streams([stream])
        raw = stream.result.text.strip()
        return self._postprocess(raw)

    def transcribe_segment(self, samples: np.ndarray) -> str:
        """Transcribe a single VAD-segmented speech chunk (for WS /ws/stt)."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self.recognizer.decode_stream(stream)
        raw = stream.result.text.strip()
        if not raw:
            return ""
        return self._postprocess(raw)
