"""Root VLA module (the single FSDP2 root): text + 2 images (+ optional state) -> Qwen3 -> per-layer K/V ->
MolmoAct2 action expert.

LLM sequence (both arms identical except how IMG tokens are produced):
    [left-pad][text tokens][IMG cam0 x64][IMG cam1 x64][STATE x1 if use_state]
Attention (bool, True = may attend): key valid AND (causal OR same image block)  -> bidirectional inside each
image (as Gemma 4's use_bidirectional_attention="vision"), causal elsewhere. Pad query rows attend to
themselves only (no all-masked rows -> no NaN); their outputs are never read (expert masks pad keys).
Position ids: cumulative count of valid tokens - 1 (pads get 0), so left padding does not shift RoPE.
"""
import math

import torch
import torch.nn as nn

from .action_expert import ActionExpert, ActionExpertConfig
from .backbone import Qwen3KVBackbone
from .vision_ef import EFVisionEmbedder


def build_attention_mask(valid: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """valid [B, L] bool; block [L] long (-1 = causal token, k>=0 = member of bidirectional block k)."""
    B, L = valid.shape
    idx = torch.arange(L, device=valid.device)
    causal = idx[None, :] <= idx[:, None]  # [Lq, Lk]
    same = (block[:, None] == block[None, :]) & (block[:, None] >= 0)
    allowed = (causal | same)[None] & valid[:, None, :]
    allowed = allowed | torch.eye(L, dtype=torch.bool, device=valid.device)[None]
    return allowed[:, None]  # [B, 1, L, L]


def position_ids_from_valid(valid: torch.Tensor) -> torch.Tensor:
    return (valid.long().cumsum(-1) - 1).clamp(min=0)


class VLA(nn.Module):
    def __init__(self, cfg, *, load_pretrained: bool = True, llm_config=None, siglip_config=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = Qwen3KVBackbone(cfg.llm_path if load_pretrained else None, config=llm_config)
        D = self.backbone.hidden_size
        if cfg.arm == "ef":
            self.vision = EFVisionEmbedder(img_size=cfg.img_size, patch=cfg.ef_patch, embed_dim=D, out_dim=D)
        else:
            from .vision_enc import ENCVision

            self.vision = ENCVision(siglip_path=cfg.siglip_path if load_pretrained else None, out_dim=D,
                                    siglip_config=siglip_config)
        self.n_img_tokens = self.vision.tokens_per_image
        self.cam_embed = nn.Parameter(torch.zeros(cfg.n_cams, D))
        self.state_proj = nn.Linear(cfg.state_dim, D) if cfg.use_state else None
        ae_cfg = ActionExpertConfig(
            hidden_size=cfg.ae_hidden, num_heads=cfg.ae_heads, num_layers=self.backbone.config.num_hidden_layers,
            mlp_ratio=cfg.ae_mlp_ratio, ffn_multiple_of=cfg.ae_ffn_multiple_of, qk_norm=cfg.ae_qk_norm,
            rope=cfg.ae_rope, context_layer_norm=cfg.ae_context_layer_norm, timestep_embed_dim=cfg.ae_timestep_dim,
            max_action_dim=cfg.action_dim, max_action_horizon=cfg.chunk,
        )
        self.action_expert = ActionExpert(ae_cfg, llm_kv_dim=self.backbone.kv_dim,
                                          llm_num_layers=self.backbone.config.num_hidden_layers)
        self.backbone.embed_tokens.weight.requires_grad_(False)  # text vocabulary is fixed (both arms)
        # The expert reads every layer's K/V, never the final hidden state. In the last layer only
        # input_layernorm/k_proj/k_norm/v_proj feed K/V; q_proj, q_norm, o_proj, the MLP and the final norm cannot
        # influence the loss (verified by tests/run_tests.py). Freeze them so no FSDP unit holds grad-less params.
        last = self.backbone.layers[-1].layer
        for p in [*last.self_attn.q_proj.parameters(), *last.self_attn.q_norm.parameters(),
                  *last.self_attn.o_proj.parameters(), *last.post_attention_layernorm.parameters(),
                  *last.mlp.parameters(), *self.backbone.norm.parameters()]:
            p.requires_grad_(False)

    # ------------------------------------------------------------------ init helpers
    @torch.no_grad()
    def text_embedding_rms(self) -> float:
        w = self.backbone.embed_tokens.weight.float()
        return float(w.pow(2).mean(-1).sqrt().median())

    @torch.no_grad()
    def init_new_modules(self, target_rms: float, generator_seed: int = 0):
        """Scale-match all inputs injected into the residual stream to the text-embedding RMS."""
        D = self.backbone.hidden_size
        self.vision.init_out(target_rms)
        nn.init.normal_(self.cam_embed, std=target_rms)
        if self.state_proj is not None:
            nn.init.normal_(self.state_proj.weight, std=target_rms / math.sqrt(self.cfg.state_dim))
            nn.init.zeros_(self.state_proj.bias)

    @torch.no_grad()
    def calibrate_vision_scale(self, images: torch.Tensor, target_rms: float) -> tuple[float, float]:
        """Rescale the vision entry's last linear so real image tokens have RMS = target (both arms)."""
        tok = self.vision(images.flatten(0, 1).to(next(self.vision.parameters()).dtype)).float()
        before = float(tok.pow(2).mean(-1).sqrt().mean())
        last = self.vision.embedding_projection if self.cfg.arm == "ef" else self.vision.connector[3]
        last.weight.mul_(target_rms / before)
        tok = self.vision(images.flatten(0, 1).to(next(self.vision.parameters()).dtype)).float()
        after = float(tok.pow(2).mean(-1).sqrt().mean())
        return before, after

    # ------------------------------------------------------------------ sequence
    def build_sequence(self, text_ids, text_valid, images, state=None):
        """text_ids/text_valid [B, Lt] (left padded); images float [B, N, 3, H, W] in [0,1]; state [B, 8]."""
        B, N = images.shape[:2]
        assert N == self.cfg.n_cams
        emb = self.backbone.embed_tokens(text_ids)
        vis = self.vision(images.flatten(0, 1).to(emb.dtype))  # [B*N, P, D]
        P = vis.shape[1]
        vis = vis.view(B, N, P, -1) + self.cam_embed.to(vis.dtype)[None, :, None, :]
        parts = [emb, vis.reshape(B, N * P, -1)]
        valid = [text_valid, torch.ones(B, N * P, dtype=torch.bool, device=text_ids.device)]
        Lt = text_ids.shape[1]
        block = [torch.full((Lt,), -1, dtype=torch.long), torch.arange(N).repeat_interleave(P)]
        if self.state_proj is not None:
            parts.append(self.state_proj(state.to(emb.dtype))[:, None])
            valid.append(torch.ones(B, 1, dtype=torch.bool, device=text_ids.device))
            block.append(torch.full((1,), -1, dtype=torch.long))
        x = torch.cat(parts, 1)
        valid = torch.cat(valid, 1)
        block = torch.cat(block).to(text_ids.device)
        return x, valid, build_attention_mask(valid, block), position_ids_from_valid(valid)

    def encode(self, text_ids, text_valid, images, state=None):
        x, valid, mask, pos = self.build_sequence(text_ids, text_valid, images, state)
        h, kvs = self.backbone(x, pos, mask)
        return h, kvs, valid

    # ------------------------------------------------------------------ training forward
    def forward(self, text_ids, text_valid, images, state, x_t, timesteps, action_valid):
        """x_t [B, F, H, A], timesteps [B, F], action_valid [B, H] bool -> predicted velocity [B, F, H, A].
        Targets are deliberately NOT inputs (FSDP mixed precision would cast them to bf16)."""
        B, F_, H, A = x_t.shape
        _, kvs, valid = self.encode(text_ids, text_valid, images, state)
        v = self.action_expert(
            x_t.reshape(B * F_, H, A), timesteps.reshape(B * F_), kv_states=kvs, encoder_valid=valid,
            action_valid=action_valid, num_expand=F_,
        )
        return v.view(B, F_, H, A)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample_actions(self, text_ids, text_valid, images, state=None, *, steps: int, noise=None):
        _, kvs, valid = self.encode(text_ids, text_valid, images, state)
        B = text_ids.shape[0]
        dtype = kvs[0][0].dtype
        x = noise if noise is not None else torch.randn(B, self.cfg.chunk, self.cfg.action_dim,
                                                        device=text_ids.device, dtype=torch.float32)
        x = x.to(dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=x.device, dtype=dtype)
            v = self.action_expert(x, t, kv_states=kvs, encoder_valid=valid, action_valid=None)
            x = x + dt * v
        return x.float()


class Tokenize:
    """Prompt -> ids, shared by training and evaluation. Left padding, no special tokens."""

    def __init__(self, llm_path: str, template: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(llm_path)
        self.template = template
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._cache = {}

    def ids(self, instruction: str) -> list[int]:
        if instruction not in self._cache:
            text = self.template.format(instruction=instruction)
            self._cache[instruction] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self._cache[instruction]

    def batch(self, instructions: list[str], device=None):
        seqs = [self.ids(s) for s in instructions]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        valid = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            valid[i, L - len(s):] = True
        return ids.to(device), valid.to(device)
