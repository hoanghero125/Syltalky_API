import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rtmlib-main"))

import config as uni_config
uni_config.mt5_path = os.path.join(HERE, "pretrained_weight", "mt5-base")

import torch
import numpy as np
import cv2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody
from models import Uni_Sign
from datasets import S2T_Dataset_online
import utils as uni_utils


class _Args:
    seed = 42
    hidden_dim = 256
    dataset = "OpenASL"    # largest English ASL dataset
    rgb_support = False
    label_smoothing = 0.2
    max_length = 256
    task = "SLT"
    online_video = ""
    output_dir = ""


class SignTranslator:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint = os.environ.get(
            "CHECKPOINT_PATH",
            os.path.join(HERE, "checkpoints", "openasl_pose_only_slt.pth"),
        )

        args = _Args()
        self.args = args
        uni_utils.set_seed(args.seed)

        print(f"[inference] device={self.device}  checkpoint={self.checkpoint}")

        self.wholebody = Wholebody(
            to_openpose=False,
            mode="lightweight",
            backend="onnxruntime",
            device=self.device,
        )

        print("[inference] Loading model...")
        self.model = Uni_Sign(args=args)

        state_dict = torch.load(self.checkpoint, map_location="cpu")["model"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        if self.device == "cuda":
            self.model.cuda()
            self.model.to(torch.bfloat16)

        print("[inference] Ready.")

    def _extract_pose(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("Video contains no frames")

        data: dict = {"keypoints": [], "scores": []}

        def _process(frame):
            frame = np.uint8(frame)
            kpts, scores = self.wholebody(frame)
            H, W, _ = frame.shape
            return kpts, scores, [W, H]

        workers = 16 if self.device == "cuda" else 4
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_process, frames))

        for kpts, scores, wh in results:
            data["keypoints"].append(kpts / np.array(wh)[None, None])
            data["scores"].append(scores)

        return data

    def translate(self, video_path: str) -> str:
        pose_data = self._extract_pose(video_path)

        online_data = S2T_Dataset_online(args=self.args)
        online_data.rgb_data = video_path
        online_data.pose_data = pose_data

        dataloader = DataLoader(
            online_data,
            batch_size=1,
            collate_fn=online_data.collate_fn,
            sampler=torch.utils.data.SequentialSampler(online_data),
        )

        use_bf16 = self.device == "cuda"

        with torch.no_grad():
            for src_input, tgt_input in dataloader:
                if use_bf16:
                    for k in src_input:
                        if isinstance(src_input[k], torch.Tensor):
                            src_input[k] = src_input[k].to(torch.bfloat16).cuda()

                stack_out = self.model(src_input, tgt_input)
                output = self.model.generate(stack_out, max_new_tokens=100, num_beams=4)

                tokenizer = self.model.mt5_tokenizer
                pad_id = tokenizer.eos_token_id
                pad_tensor = torch.ones(150 - len(output[0])) * pad_id
                output[0] = torch.cat((output[0], pad_tensor.long()), dim=0)
                output = pad_sequence(output, batch_first=True, padding_value=pad_id)
                return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return ""
