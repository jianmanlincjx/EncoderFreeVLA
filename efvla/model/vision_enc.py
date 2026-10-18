"""Conventional VLA vision entry: pretrained SigLIP2 so400m-patch16-256 + 2x2 spatial merge + MLP connector.

256x256 input -> 16x16 patch tokens (1152-d, last_hidden_state already includes post_layernorm)
-> 2x2 merge -> 8x8 tokens of 4608-d -> LN -> Linear -> GELU -> Linear -> LLM width.
Same token count per image (64) as the EF arm.
"""
import math

import torch
import torch.nn as nn


def merge_2x2(x: torch.Tensor, grid: int) -> torch.Tensor:
    """[B, grid*grid, C] -> [B, (grid/2)^2, 4C]; merged token (i, j) holds positions (2i..2i+1, 2j..2j+1)."""
    B, N, C = x.shape
    assert N == grid * grid and grid % 2 == 0
    g = grid // 2
    x = x.view(B, g, 2, g, 2, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, g * g, 4 * C)


class ENCVision(nn.Module):
    def __init__(self, *, siglip_path: str | None, out_dim: int, siglip_config=None):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel

        if siglip_path is not None:
            model, info = SiglipVisionModel.from_pretrained(
                siglip_path, dtype=torch.float32, vision_use_head=False, output_loading_info=True
            )
            assert not info["missing_keys"], f"SigLIP missing keys: {info['missing_keys']}"
            bad = [k for k in info.get("mismatched_keys", [])]
            assert not bad, f"SigLIP mismatched keys: {bad}"
            self.loading_info = {k: list(v) for k, v in info.items()}
        else:  # build from config (eval/tests); weights come from a checkpoint
            model = SiglipVisionModel(siglip_config)
            self.loading_info = None
        cfg = model.config
        assert cfg.model_type == "siglip_vision_model", cfg.model_type  # not the NaFlex siglip2 architecture
        assert not getattr(cfg, "vision_use_head", False)
        # wrap each encoder layer so activation checkpointing sits inside the FSDP unit (same keys in train/eval)
        from .ckpt_wrap import CkptWrap

        enc = model.vision_model.encoder
        enc.layers = nn.ModuleList([CkptWrap(l) for l in enc.layers])
        self.siglip = model
        self.image_size = cfg.image_size
        self.grid = cfg.image_size // cfg.patch_size
        c = cfg.hidden_size
        self.connector = nn.Sequential(
            nn.LayerNorm(4 * c), nn.Linear(4 * c, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def init_out(self, target_rms: float):
        """GELU output RMS is data dependent; scale the last layer from a measured RMS later if needed."""
        last = self.connector[3]
        nn.init.normal_(last.weight, std=target_rms / math.sqrt(last.in_features))
        nn.init.zeros_(last.bias)

    def set_grad_ckpt(self, on: bool):
        for l in self.siglip.vision_model.encoder.layers:
            l.grad_ckpt = on

    @property
    def tokens_per_image(self) -> int:
        return (self.grid // 2) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: float [B, 3, 256, 256] in [0, 1] -> [B, 64, out_dim]."""
        assert images.shape[-1] == self.image_size, images.shape
        x = (images - 0.5) / 0.5  # SigLIP processor: rescale 1/255 then normalise mean=std=0.5
        x = x.to(next(self.siglip.parameters()).dtype)
        h = self.siglip(pixel_values=x, interpolate_pos_encoding=False).last_hidden_state
        return self.connector(merge_2x2(h, self.grid))
