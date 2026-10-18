"""Qwen3 backbone with an explicit layer loop that also returns every layer's K/V (after k_norm and RoPE),
which is exactly what MolmoAct2's action expert reads from the KV cache.

Why not call HF Qwen3Model.forward: we need per-layer K/V *with gradients* during training together with
gradient checkpointing (HF disables the cache under checkpointing), and with FSDP2 every sharded unit must be
entered through its own forward. Each HF Qwen3DecoderLayer is therefore wrapped in KVDecoderLayer, whose
forward re-implements Qwen3DecoderLayer.forward/Qwen3Attention.forward with the same submodules.
Equivalence with HF (hidden states and cache K/V) is checked in tests/run_tests.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class KVDecoderLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer  # transformers Qwen3DecoderLayer (weights live here)
        self.grad_ckpt = False

    def forward(self, h, cos, sin, attn_mask):
        # activation checkpointing lives *inside* the (FSDP-sharded) unit, as in torchtitan
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, h, cos, sin, attn_mask, use_reentrant=False)
        return self._forward(h, cos, sin, attn_mask)

    def _forward(self, h, cos, sin, attn_mask):
        layer, attn = self.layer, self.layer.self_attn
        B, L, _ = h.shape
        hd = attn.head_dim
        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_norm(attn.q_proj(x).view(B, L, -1, hd)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(B, L, -1, hd)).transpose(1, 2)
        v = attn.v_proj(x).view(B, L, -1, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=attn.scaling, enable_gqa=True
        )
        o = o.transpose(1, 2).reshape(B, L, -1)
        h = residual + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v  # k, v: [B, n_kv_heads, L, head_dim]


class Qwen3KVBackbone(nn.Module):
    def __init__(self, llm_path: str | None, *, config=None, model=None, dtype=torch.float32):
        super().__init__()
        from transformers import Qwen3ForCausalLM, Qwen3Model

        if model is not None:  # wrap an existing Qwen3Model (tests share weights with HF this way)
            pass
        elif llm_path is not None:
            lm = Qwen3ForCausalLM.from_pretrained(llm_path, dtype=dtype, attn_implementation="sdpa")
            model = lm.model
            del lm
        else:
            model = Qwen3Model(config)
        cfg = model.config
        assert not getattr(cfg, "use_sliding_window", False), "sliding-window layers are not supported here"
        assert getattr(cfg, "layer_types", None) in (None,) or all(t == "full_attention" for t in cfg.layer_types)
        self.config = cfg
        self.embed_tokens = model.embed_tokens
        self.layers = nn.ModuleList([KVDecoderLayer(l) for l in model.layers])
        self.norm = model.norm
        self.rotary_emb = model.rotary_emb
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

    def set_grad_ckpt(self, on: bool):
        for l in self.layers:
            l.grad_ckpt = on

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def forward(self, inputs_embeds, position_ids, attn_mask):
        """inputs_embeds [B,L,D]; position_ids [B,L] long; attn_mask bool [B,1,L,L] (True = may attend).
        Returns final hidden (after norm) and a list of (K, V) per layer, each [B, L, n_kv*head_dim]."""
        assert attn_mask.dtype == torch.bool and attn_mask.dim() == 4, (attn_mask.dtype, attn_mask.shape)
        assert position_ids.dtype == torch.long
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        kvs = []
        for layer in self.layers:
            h, k, v = layer(h, cos, sin, attn_mask)
            B, H, L, D = k.shape
            kvs.append((k.transpose(1, 2).reshape(B, L, H * D), v.transpose(1, 2).reshape(B, L, H * D)))
        return self.norm(h), kvs
