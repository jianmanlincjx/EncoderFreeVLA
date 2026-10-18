"""MolmoAct2 action expert, ported line-for-line from the official HF remote code
(transformers_modules/MolmoAct2/modeling_molmoact2.py, lines 90-111 helpers, 130-860 modules) and the
LeRobot training path (lerobot/policies/molmoact2/modeling_molmoact2.py, lines 587-607 time sampling,
2485-2540 flow targets, 2684-2770 + 2903-2921 per-layer KV conditioning, 2975-2990 loss).
Module/parameter names are kept identical so the original weights load with load_state_dict(strict=True);
tests/run_tests.py checks numerical equality against the original class.

Removed relative to the original (not used here): modulation caches for CUDA graphs, discrete-state guards.
Flow convention (MolmoAct2): t=0 noise, t=1 data; x_t=(1-t)*noise + t*action; target velocity = action - noise;
inference integrates dx/dt = v from t=0 to 1 with Euler steps.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F


@dataclass
class ActionExpertConfig:
    hidden_size: int = 768
    num_heads: int = 8
    num_layers: int = 36
    mlp_ratio: float = 4.0
    ffn_multiple_of: int = 256
    attn_dropout: float = 0.0
    dropout: float = 0.0
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    context_layer_norm: bool = True
    timestep_embed_dim: int = 256
    causal_attn: bool = False
    max_action_dim: int = 7
    max_action_horizon: int = 10


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ActionExpertRMSNorm(nn.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = (x_float * torch.rsqrt(variance + self.eps)).to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def build_cache(self, *, seq_len, device, dtype):
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / max(half_dim, 1)))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        return cos, sin

    def forward(self, q, k, *, rope_cache=None):
        if rope_cache is None:
            rope_cache = self.build_cache(seq_len=q.shape[-2], device=q.device, dtype=q.dtype)
        cos, sin = rope_cache
        half_dim = self.head_dim // 2

        def _apply(x):
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class ActionExpertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, attn_dropout=0.0, proj_dropout=0.0, qk_norm=True,
                 qk_norm_eps=1e-6, use_rope=True) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q, k):
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(self, q, k, v, *, attn_mask=None, is_causal=False):
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        return out.transpose(1, 2).contiguous()

    def forward(self, x, *, attn_mask=None, is_causal=False, rope_cache=None):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].contiguous()
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cache=rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, attn_dropout=0.0, proj_dropout=0.0, qk_norm=True,
                 qk_norm_eps=1e-6) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _as_heads(self, x):
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(self, q, k, v, *, attn_mask=None):
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)
        return out.transpose(1, 2).contiguous()

    def forward(self, x, *, kv_k, kv_v, attn_mask=None):
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self._as_heads(kv_k)
        v = self._as_heads(kv_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertMLP(nn.Module):
    def __init__(self, hidden_size, *, mlp_ratio, multiple_of, dropout=0.0) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning):
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, *, mlp_ratio, ffn_multiple_of, attn_dropout=0.0, dropout=0.0,
                 qk_norm=True, qk_norm_eps=1e-6, rope=True) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(hidden_size, num_heads, attn_dropout=attn_dropout,
                                                   proj_dropout=dropout, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps,
                                                   use_rope=rope)
        self.cross_attn = ActionExpertCrossAttention(hidden_size, num_heads, attn_dropout=attn_dropout,
                                                     proj_dropout=dropout, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps)
        self.mlp = ActionExpertMLP(hidden_size, mlp_ratio=mlp_ratio, multiple_of=ffn_multiple_of, dropout=dropout)
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(self, x, conditioning, *, cross_kv, self_attn_mask=None, attn_mask=None, is_causal=False,
                modulation=None, rope_cache=None):
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp) = modulation
        x = x + gate_msa.unsqueeze(1) * self.self_attn(_modulate(self.self_norm(x), shift_msa, scale_msa),
                                                       attn_mask=self_attn_mask, is_causal=is_causal,
                                                       rope_cache=rope_cache)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(_modulate(self.cross_norm(x), shift_mca, scale_mca),
                                                        kv_k=cross_kv[0], kv_v=cross_kv[1], attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.ff_norm(x), shift_mlp, scale_mlp))
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, conditioning, *, modulation=None):
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(2, dim=1)
        shift, scale = modulation
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        half_dim = self.dim // 2
        freq = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
                         * (-math.log(10000.0) / max(half_dim - 1, 1)))
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _expand(x: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
    """MolmoAct2 `_expand_mask`: [B, ...] -> [B*n, ...], batch-major (b0 f0, b0 f1, ...)."""
    if x is None or n == 1:
        return x
    return x.unsqueeze(1).expand(-1, n, *([-1] * (x.ndim - 1))).reshape(x.shape[0] * n, *x.shape[1:])


class ActionExpert(nn.Module):
    def __init__(self, config: ActionExpertConfig, *, llm_kv_dim: int, llm_num_layers: int):
        super().__init__()
        if config.num_layers != llm_num_layers:
            raise ValueError(f"one action block per LLM layer required ({config.num_layers} vs {llm_num_layers})")
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size)
        self.context_k_proj = nn.Linear(llm_kv_dim, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(llm_kv_dim, config.hidden_size, bias=False)
        self.context_norm = (ActionExpertRMSNorm(config.hidden_size, eps=1e-6) if config.context_layer_norm
                             else nn.Identity())
        self.blocks = nn.ModuleList([
            ActionExpertBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio,
                              ffn_multiple_of=config.ffn_multiple_of, attn_dropout=config.attn_dropout,
                              dropout=config.dropout, qk_norm=config.qk_norm, qk_norm_eps=config.qk_norm_eps,
                              rope=config.rope)
            for _ in range(config.num_layers)
        ])
        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.context_k_proj)
        _init_linear(self.context_v_proj)
        if isinstance(self.context_norm, ActionExpertRMSNorm):
            self.context_norm.reset_parameters()
        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
            for n in (block.self_norm, block.cross_norm, block.ff_norm, block.self_attn.q_norm,
                      block.self_attn.k_norm, block.cross_attn.q_norm, block.cross_attn.k_norm):
                if n is not None:
                    n.reset_parameters()
        self.final_layer.norm.reset_parameters()
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    # ---- context preparation (original: _project_kv_tensor / _prepare_kv_context / mask builders) ----
    def _reshape_hidden_to_heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _time_conditioning(self, timesteps):
        conditioning = self.time_embed[0](timesteps)
        first_linear = self.time_embed[1]
        conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _project_kv_tensor(self, x, proj):
        return self._reshape_hidden_to_heads(self.context_norm(proj(x)))

    def prepare_kv_contexts(self, kv_states):
        assert len(kv_states) == len(self.blocks), (len(kv_states), len(self.blocks))
        out = []
        for block, (k_in, v_in) in zip(self.blocks, kv_states):
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            if block.cross_attn.k_norm is not None:
                k_ctx = block.cross_attn.k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            out.append((k_ctx, v_ctx))
        return out

    @staticmethod
    def build_cross_mask(encoder_valid: torch.Tensor, dtype) -> torch.Tensor:
        """Original _build_cross_attention_mask: additive float mask, 0 for valid keys, finfo.min otherwise."""
        mask = encoder_valid[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def build_self_mask(self, action_valid: Optional[torch.Tensor], seq_len, device, dtype):
        mask = None
        if action_valid is not None:
            key_mask = (~action_valid.to(device=device, dtype=torch.bool))[:, None, None, :].to(dtype=dtype)
            mask = key_mask * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
            causal = causal[None, None].to(dtype=dtype) * torch.finfo(dtype).min
            mask = causal if mask is None else mask + causal
        return mask

    def forward(self, x_t, timesteps, *, kv_states, encoder_valid, action_valid=None, num_expand: int = 1):
        """Predict velocity.
        x_t: [B*num_expand, H, A]; timesteps: [B*num_expand]; kv_states: per layer (K, V) with K,V [B, L, kv_dim];
        encoder_valid: [B, L] bool (keys the expert may read); action_valid: [B, H] bool or None.
        num_expand > 1 reuses one LLM forward for several flow timesteps (MolmoAct2 num_flow_timesteps)."""
        dtype = x_t.dtype
        seq_len = x_t.shape[1]
        if seq_len > self.config.max_action_horizon:
            raise ValueError(f"horizon {seq_len} > max_action_horizon {self.config.max_action_horizon}")
        kv_contexts = [(_expand(k, num_expand), _expand(v, num_expand)) for k, v in self.prepare_kv_contexts(kv_states)]
        cross_mask = _expand(self.build_cross_mask(encoder_valid, dtype), num_expand)
        self_mask = _expand(self.build_self_mask(action_valid, seq_len, x_t.device, dtype), num_expand)
        valid = None
        if action_valid is not None:
            valid = _expand(action_valid.to(dtype=dtype).unsqueeze(-1), num_expand)
        rope_cache = None
        if self.blocks[0].self_attn.rope is not None:
            rope_cache = self.blocks[0].self_attn.rope.build_cache(seq_len=seq_len, device=x_t.device, dtype=dtype)
        conditioning = self._time_conditioning(timesteps)
        x = self.action_embed(x_t)
        if valid is not None:
            x = x * valid
        for block, kv in zip(self.blocks, kv_contexts):
            x = block(x, conditioning, cross_kv=kv, self_attn_mask=self_mask, attn_mask=cross_mask,
                      is_causal=self.config.causal_attn, modulation=None, rope_cache=rope_cache)
            if valid is not None:
                x = x * valid
        out = self.final_layer(x, conditioning)
        if valid is not None:
            out = out * valid
        return out


def sample_beta_timesteps(n: int, *, device, alpha: float, beta: float, cutoff: float, time_offset: float,
                          time_scale: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """MolmoAct2 _sample_beta_timesteps: t = offset + (min(cutoff, offset+scale) - offset) * Beta(alpha, beta)."""
    if cutoff < time_offset:
        raise ValueError("cutoff must be >= time_offset")
    upper = min(cutoff, time_offset + time_scale)
    scale = upper - time_offset
    if generator is None:
        samples = Beta(torch.tensor(alpha, device=device), torch.tensor(beta, device=device)).sample((n,))
    else:  # reproducible path (tests / diagnostics): Beta via two Gammas from a seeded generator is not
        # available in torch, so draw on CPU with a forked RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (), generator=generator)))
            samples = Beta(torch.tensor(alpha), torch.tensor(beta)).sample((n,)).to(device)
    if scale <= 0:
        return torch.full((n,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def flow_targets(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    """actions [B,H,A]; noise [B,F,H,A]; t [B,F] -> x_t [B,F,H,A], target velocity [B,F,H,A]."""
    tb = t[:, :, None, None]
    a = actions.unsqueeze(1).expand_as(noise)
    return (1.0 - tb) * noise + tb * a, a - noise
