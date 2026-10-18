"""Load a training checkpoint into a plain (non-FSDP) model and expose act() for closed-loop control."""
import numpy as np
import torch

from efvla.config import config_from_dict
from efvla.data.normalize import Normalizer, binarize_gripper
from efvla.data.transforms import preprocess
from efvla.model.vla import VLA, Tokenize


def load_model(ckpt_path: str, device, dtype=torch.bfloat16):
    from transformers import Qwen3Config, SiglipVisionConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    llm_cfg = Qwen3Config.from_dict(ck["llm_config"])
    sig_cfg = SiglipVisionConfig.from_dict(ck["siglip_config"]) if ck.get("siglip_config") else None
    model = VLA(cfg, load_pretrained=False, llm_config=llm_cfg, siglip_config=sig_cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    return model, cfg, ck


class Policy:
    def __init__(self, ckpt_path: str, device="cuda", dtype=torch.bfloat16, blank_images: bool = False):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model, self.cfg, ck = load_model(ckpt_path, self.device, dtype)
        self.norm = Normalizer(ck["stats"])
        self.tasks = ck["tasks"]
        self.step = ck["step"]
        self.tok = Tokenize(self.cfg.llm_path, self.cfg.prompt_template)
        self.blank = blank_images
        self.gen = torch.Generator(device=self.device)

    def reseed(self, seed: int):
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def act(self, images_u8: np.ndarray, state8: np.ndarray, instruction: str) -> np.ndarray:
        """images_u8 [2, 256, 256, 3] uint8 in dataset orientation (agentview, wrist); returns [H, 7] env actions."""
        x = torch.from_numpy(np.ascontiguousarray(images_u8))[None]
        x = preprocess(x, self.cfg.model_img_size).to(self.device)  # MolmoAct2 eval path (uint8 resize on CPU)
        if self.blank:
            x = torch.zeros_like(x)
        ids, val = self.tok.batch([instruction], self.device)
        st = torch.from_numpy(self.norm.norm_state(state8))[None].to(self.device, self.dtype)
        noise = torch.randn(1, self.cfg.chunk, self.cfg.action_dim, device=self.device, generator=self.gen)
        a = self.model.sample_actions(ids, val, x.to(self.dtype), st, steps=self.cfg.flow_infer_steps, noise=noise)
        a = self.norm.unnorm_actions(a[0].cpu().numpy())
        return binarize_gripper(a)
