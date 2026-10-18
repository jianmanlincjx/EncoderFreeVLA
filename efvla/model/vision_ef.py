"""Encoder-free vision entry: a port of transformers' Gemma4UnifiedVisionEmbedder
(transformers/models/gemma4_unified/modeling_gemma4_unified.py), with the patch size as a parameter.

Gemma4UnifiedVisionEmbedder:  raw_patches -> LN1 -> Dense -> LN2 -> + factorised (x, y) pos-emb -> LN3
                              -> RMSNorm(no scale) -> Linear(no bias)  [Gemma4UnifiedMultimodalEmbedder]
Differences from Gemma 4 12B (documented on purpose):
  * input = MolmoAct2's 378x378 image, patch 27 px -> 14x14 = 196 tokens (MolmoAct2's pooled grid);
    Gemma uses 48 px (= 16 px x 3x3 merge).
  * fixed 14x14 grid, so the position table has 14 rows per axis instead of 1120.
  * the last Linear is rescaled so image tokens match the LLM's text-embedding RMS (VLA.calibrate_vision_scale).
Pixels arrive in [-1, 1] (MolmoAct2 convention). Gemma feeds [0, 1]; the first op is a per-patch LayerNorm,
which is invariant to the affine map x -> 2x - 1 (up to eps), so the two conventions are equivalent.
Patch layout inside a flattened patch is (ph, pw, c), as produced by Gemma's patchifier.
"""
import math

import torch
import torch.nn as nn


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """x [B,3,H,W] -> [B, (H/p)*(W/p), p*p*3], patches in row-major order, inner layout (ph, pw, c)."""
    B, C, H, W = x.shape
    nh, nw = H // p, W // p
    x = x.reshape(B, C, nh, p, nw, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, nh * nw, p * p * C)


def grid_positions(nh: int, nw: int, device=None) -> torch.Tensor:
    """[nh*nw, 2] integer (x, y) = (column, row), matching Gemma's image_position_ids convention."""
    rows = torch.arange(nh, device=device).repeat_interleave(nw)
    cols = torch.arange(nw, device=device).repeat(nh)
    return torch.stack([cols, rows], -1)


class RMSNormNoScale(nn.Module):
    """Gemma4UnifiedRMSNorm(with_scale=False): x * rsqrt(mean(x^2) + eps), computed in fp32."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype)


class EFVisionEmbedder(nn.Module):
    def __init__(self, *, img_size: int, patch: int, embed_dim: int, out_dim: int, rms_eps: float = 1e-6,
                 pos_init_std: float = 0.02):
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        patch_dim = patch * patch * 3
        # nn.LayerNorm default eps (1e-5) and affine=True, exactly as in Gemma4UnifiedVisionEmbedder
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)
        self.pos_embedding = nn.Parameter(torch.empty(self.grid, 2, embed_dim))
        self.pos_norm = nn.LayerNorm(embed_dim)
        self.pre_proj_norm = RMSNormNoScale(rms_eps)
        self.embedding_projection = nn.Linear(embed_dim, out_dim, bias=False)
        nn.init.normal_(self.pos_embedding, std=pos_init_std)
        nn.init.normal_(self.patch_dense.weight, std=0.02)
        nn.init.zeros_(self.patch_dense.bias)
        self.register_buffer("positions", grid_positions(self.grid, self.grid), persistent=False)

    def init_out(self, target_rms: float):
        """RMSNorm output has unit RMS; a Linear(D->D) with N(0, s^2) weights gives output RMS ~ s*sqrt(D)."""
        nn.init.normal_(self.embedding_projection.weight, std=target_rms / math.sqrt(self.embedding_projection.in_features))

    @property
    def last_linear(self):
        return self.embedding_projection

    @property
    def tokens_per_image(self) -> int:
        return self.grid * self.grid

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: float [B, 3, H, W] (MolmoAct2 convention [-1, 1]) -> [B, grid*grid, out_dim]."""
        x = patchify(images, self.patch).to(self.patch_dense.weight.dtype)
        x = self.patch_ln1(x)
        x = self.patch_dense(x)
        x = self.patch_ln2(x)
        pos = self.positions  # [P, 2] (x, y)
        axes = torch.arange(2, device=pos.device)
        x = x + self.pos_embedding[pos, axes].sum(-2)  # table[x, 0] + table[y, 1]
        x = self.pos_norm(x)
        return self.embedding_projection(self.pre_proj_norm(x))
