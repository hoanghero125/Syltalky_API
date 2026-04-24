import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get(
    "STT_MODEL_DIR",
    os.path.join(HERE, "model"),
)


class SpeechTranscriber:
    def __init__(self):
        import sherpa_onnx

        encoder = os.path.join(MODEL_DIR, "encoder-epoch-20-avg-10.int8.onnx")
        decoder = os.path.join(MODEL_DIR, "decoder-epoch-20-avg-10.int8.onnx")
        joiner  = os.path.join(MODEL_DIR, "joiner-epoch-20-avg-10.int8.onnx")
        bpe     = os.path.join(MODEL_DIR, "bpe.model")

        print("[stt] Loading Zipformer RNNT...")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            bpe_model=bpe,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
            provider="cuda",
        )

        print("[stt] Ready.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy float32 audio array to Vietnamese text.
        Audio should be mono, float32, values in [-1, 1].
        """
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)

        # signal end of audio
        tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()

        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

        return self.recognizer.get_result(stream).text.strip()
