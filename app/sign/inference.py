import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "rtmlib-main"))

import config as uni_config
uni_config.mt5_path = os.path.join(HERE, "pretrained_weight", "mt5-base")

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from rtmlib import Wholebody
from models import Uni_Sign
from datasets import S2T_Dataset_online
import utils as uni_utils


class _Args:
    seed = 42
    hidden_dim = 256
    dataset = "OpenASL"
    rgb_support = False
    label_smoothing = 0.2
    max_length = 256
    task = "SLT"
    online_video = ""
    output_dir = ""


class SignTranslator:
    def __init__(self):
        self.args = _Args()
        uni_utils.set_seed(self.args.seed)

        print("[inference] Loading RTMPose...")
        self.wholebody = Wholebody(
            to_openpose=False,
            mode="lightweight",
            backend="onnxruntime",
            device="cuda",
        )

        checkpoint = os.environ.get(
            "CHECKPOINT_PATH",
            os.path.join(HERE, "checkpoints", "openasl_pose_only_slt.pth"),
        )

        print(f"[inference] Loading model from {checkpoint}...")
        self.model = Uni_Sign(args=self.args)
        state_dict = torch.load(checkpoint, map_location="cpu")["model"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.cuda()
        self.model.to(torch.bfloat16)

        print("[inference] Ready.")

    def extract_keypoints(self, frame: np.ndarray) -> tuple:
        """Extract keypoints from a single BGR frame. Returns (keypoints_norm, scores)."""
        frame = np.uint8(frame)
        kpts, scores = self.wholebody(frame)
        H, W, _ = frame.shape
        kpts_norm = kpts / np.array([W, H])[None, None]
        return kpts_norm, scores

    def translate(self, pose_data: dict) -> str:
        """Translate accumulated pose_data dict {keypoints: [...], scores: [...]} to English."""
        online_data = S2T_Dataset_online(args=self.args)
        online_data.rgb_data = ""
        online_data.pose_data = pose_data

        dataloader = DataLoader(
            online_data,
            batch_size=1,
            collate_fn=online_data.collate_fn,
            sampler=torch.utils.data.SequentialSampler(online_data),
        )

        with torch.no_grad():
            for src_input, tgt_input in dataloader:
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
