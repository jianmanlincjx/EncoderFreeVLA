"""Conventional-VLA vision entry, identical in structure to MolmoAct2's vision front end.

ViT: SigLIP2 so400m-patch14-384 (same dims as MolmoAct2's ViT: 1152 wide, 27 layers, 16 heads, FFN 4304,
patch 14). At 378x378 it yields 27x27 = 729 patches, exactly its 729 learned positions (no interpolation).
Adapter (ported from Checkpoint/MolmoAct2/modeling_molmoact2.py, MolmoAct2VisionBackbone, lines 1457-1928):
  * features of ViT blocks vit_layers=(-3, -9) -> blocks 24 and 18 (0-based), concatenated in that order (2304-d);
    only the first 25 blocks are kept (MolmoAct2 truncates the same way); SigLIP's post_layernorm is not used.
  * 2x2 attention pooling on the 27x27 grid padded to 28x28 at the bottom/right (padding masked):
    query = mean of the valid window members, keys/values = the window (ViTMultiHeadDotProductAttention,
    16 heads x 72, input 2304 -> 1152, float32 attention) -> 14x14 = 196 tokens.
  * ImageProjectorMLP (SwiGLU, no bias): 1152 -> 9728 -> LLM width.
Module/parameter names of the adapter match MolmoAct2 (image_pooling_2d.{wq,wk,wv,wo}, image_projector.{w1,w2,w3}).
The adapter is randomly initialised (normal 0.02, zero bias), the ViT starts from the SigLIP2 checkpoint.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ckpt_wrap import CkptWrap


class ViTMultiHeadDotProductAttention(nn.Module):
    """MolmoAct2 ViTMultiHeadDotProductAttention, sdpa path (num_heads == num_kv_heads here)."""

    def __init__(self, hidden_size, num_heads, num_key_value_heads, head_dim, use_bias=True, input_dim=None,
                 float32_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        input_dim = input_dim or hidden_size
        self.wq = nn.Linear(input_dim, num_heads * head_dim, bias=use_bias)
        self.wk = nn.Linear(input_dim, num_key_value_heads * head_dim, bias=use_bias)
        self.wv = nn.Linear(input_dim, num_key_value_heads * head_dim, bias=use_bias)
        self.wo = nn.Linear(num_heads * head_dim, hidden_size)
        self.float32_attention = float32_attention

    def _split_heads(self, x, n):
        return x.reshape(x.shape[:2] + (n, self.head_dim))

    def forward(self, inputs_q, inputs_kv, attn_mask=None):
        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_kv), self.wv(inputs_kv)
        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)
        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
        og_dtype = xq.dtype
        if self.float32_attention:
            xq, xk, xv = xq.float(), xk.float(), xv.float()
        out = F.scaled_dot_product_attention(xq.transpose(1, 2).contiguous(), xk.transpose(1, 2).contiguous(),
                                             xv.transpose(1, 2).contiguous(), attn_mask=attn_mask, is_causal=False,
                                             dropout_p=0.0).transpose(1, 2)
        out = out.to(og_dtype).reshape(out.shape[:2] + (self.hidden_size,))
        return self.wo(out)


class ImageProjectorMLP(nn.Module):
    """MolmoAct2 ImageProjectorMLP: w2(silu(w1 x) * w3 x), no bias."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def pooling_windows(grid: int, pool: int, device=None):
    """Window membership as MolmoAct2 arange_for_pooling: pad (pool*ceil(g/pool) - g) split as
    [pad//2 before, (pad+1)//2 after]; returns patch index [n_win, pool*pool] with -1 for padding."""
    idx = torch.arange(grid * grid, device=device).view(grid, grid)
    g2 = (grid + pool - 1) // pool
    pad = pool * g2 - grid
    idx = F.pad(idx, (pad // 2, (pad + 1) // 2, pad // 2, (pad + 1) // 2), value=-1)
    return idx.view(g2, pool, g2, pool).permute(0, 2, 1, 3).reshape(g2 * g2, pool * pool)


class ENCVision(nn.Module):
    def __init__(self, *, siglip_path: str | None, out_dim: int, siglip_config=None, img_size: int = 378,
                 vit_layers=(-3, -9), pool: int = 2, adapter_hidden: int = 1152, adapter_heads: int = 16,
                 adapter_head_dim: int = 72, adapter_intermediate: int = 9728):
        super().__init__()
        from transformers import SiglipVisionConfig, SiglipVisionModel

        if siglip_path is not None:
            vcfg = SiglipVisionConfig.from_pretrained(siglip_path)
            vcfg.vision_use_head = False
            model, info = SiglipVisionModel.from_pretrained(siglip_path, config=vcfg, dtype=torch.float32,
                                                            output_loading_info=True)
            assert not info["missing_keys"], f"SigLIP missing keys: {info['missing_keys']}"
            stray = [k for k in info["unexpected_keys"] if k.startswith("vision_model.") and not k.startswith("vision_model.head.")]
            assert not stray, f"unexpected vision-tower keys: {stray[:5]}"
            self.loading_info = {k: list(v) for k, v in info.items()}
        else:
            model = SiglipVisionModel(siglip_config)
            self.loading_info = None
        cfg = model.config
        assert cfg.model_type == "siglip_vision_model", cfg.model_type
        n = cfg.num_hidden_layers
        self.vit_layers = [l if l >= 0 else l + n for l in vit_layers]
        keep = max(self.vit_layers) + 1
        vm = model.vision_model
        vm.encoder.layers = nn.ModuleList([CkptWrap(l) for l in list(vm.encoder.layers)[:keep]])
        vm.post_layernorm = nn.Identity()  # not used (features come from intermediate blocks)
        self.siglip = model
        self.patch = cfg.patch_size
        self.img_size = img_size
        self.grid = img_size // cfg.patch_size
        assert self.grid * self.grid == vm.embeddings.num_patches, \
            f"{img_size}px gives {self.grid}^2 patches but the ViT has {vm.embeddings.num_patches} positions"
        self.pool = pool
        self.register_buffer("pool_idx", pooling_windows(self.grid, pool), persistent=False)
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=adapter_hidden, num_heads=adapter_heads, num_key_value_heads=adapter_heads,
            head_dim=adapter_head_dim, input_dim=cfg.hidden_size * len(self.vit_layers), float32_attention=True)
        self.image_projector = ImageProjectorMLP(adapter_hidden, adapter_intermediate, out_dim)
        for m in [*self.image_pooling_2d.modules(), *self.image_projector.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_grad_ckpt(self, on: bool):
        for l in self.siglip.vision_model.encoder.layers:
            l.grad_ckpt = on

    def init_out(self, target_rms: float):
        pass  # set exactly by VLA.calibrate_vision_scale on real images

    @property
    def last_linear(self):
        return self.image_projector.w2

    @property
    def tokens_per_image(self) -> int:
        return self.pool_idx.shape[0]

    def vit_features(self, x):
        """x: [B, 3, H, W] in [-1, 1] -> concat of the selected block outputs [B, grid^2, C*len(vit_layers)]."""
        vm = self.siglip.vision_model
        h = vm.embeddings(x.to(vm.embeddings.patch_embedding.weight.dtype), interpolate_pos_encoding=False)
        outs = {}
        for i, layer in enumerate(vm.encoder.layers):
            h = layer(h, None)
            if isinstance(h, tuple):
                h = h[0]
            if i in self.vit_layers:
                outs[i] = h
        return torch.cat([outs[l] for l in self.vit_layers], -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: float [B, 3, 378, 378] in [-1, 1] -> [B, 196, out_dim]."""
        assert images.shape[-1] == self.img_size and images.shape[-2] == self.img_size, images.shape
        feats = self.vit_features(images)
        B, N, C = feats.shape
        idx = self.pool_idx  # [n_win, pool^2]
        valid = idx >= 0
        to_pool = feats[:, idx.clamp(min=0)]  # [B, n_win, pool^2, C]
        to_pool = to_pool * valid.to(to_pool.dtype)[None, :, :, None]
        to_pool = to_pool.reshape(-1, idx.shape[1], C)
        v = valid[None].expand(B, -1, -1).reshape(-1, idx.shape[1])
        denom = v.float().sum(-1).clamp(min=1.0)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
        pooled = self.image_pooling_2d(query, to_pool, attn_mask=v[:, None, None, :])
        pooled = pooled.reshape(B, idx.shape[0], -1)
        return self.image_projector(pooled)
