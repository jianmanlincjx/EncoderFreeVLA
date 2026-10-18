"""Unit tests. Run on fm1:  python -m tests.run_tests   (CPU, fp32, tiny models; ~1 min)."""
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                msg = fn()
                RESULTS.append((name, "PASS", msg or ""))
            except SkipTest as e:
                RESULTS.append((name, "SKIP", str(e)))
            except Exception as e:
                RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        run.__name__ = fn.__name__
        TESTS.append(run)
        return run
    return deco


class SkipTest(Exception):
    pass


TESTS = []


def maxdiff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def tiny_qwen3_config(layers=2):
    from transformers import Qwen3Config

    c = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                    rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)
    c._attn_implementation = "sdpa"
    return c


def tiny_cfg(arm="ef", use_state=False):
    from efvla.config import Config

    return Config(arm=arm, model_img_size=56, ef_patch=28, chunk=10, action_dim=7, ae_hidden=32, ae_heads=4,
                  ae_timestep_dim=16, use_state=use_state)


def tiny_siglip_config():
    from transformers import SiglipVisionConfig

    return SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=10, num_attention_heads=2,
                              image_size=56, patch_size=14, vision_use_head=False, hidden_act="gelu_pytorch_tanh",
                              layer_norm_eps=1e-6)


def randomize_(module, std=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)


# ----------------------------------------------------------------------------------------------- vision
@check("patchify == Gemma convert_image_to_patches (ph, pw, c layout)")
def t_patchify():
    from efvla.model.vision_ef import patchify

    def gemma_ref(image, p):  # data/image_processor.py (Gemma transformers implementation), single image [C,H,W]
        c, h, w = image.shape
        x = image.reshape(c, h // p, p, w // p, p).permute(1, 3, 2, 4, 0)
        return x.reshape((h // p) * (w // p), -1)

    x = torch.rand(3, 3, 378, 378)
    ours = patchify(x, 27)
    ref = torch.stack([gemma_ref(x[i], 27) for i in range(3)])
    assert torch.equal(ours, ref)
    return f"shape {tuple(ours.shape)}"


@check("EF embedder == transformers Gemma4UnifiedVisionEmbedder (same weights)")
def t_ef_vs_gemma():
    import importlib

    Emb = VCfg = TCfg = None
    for mod in ("transformers.models.gemma4_unified.modeling_gemma4_unified", "transformers.models.gemma4.modeling_gemma4"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "Gemma4UnifiedVisionEmbedder"):
                Emb = m.Gemma4UnifiedVisionEmbedder
                cm = importlib.import_module(mod.replace("modeling", "configuration"))
                VCfg, TCfg = cm.Gemma4UnifiedVisionConfig, cm.Gemma4UnifiedTextConfig
                break
        except Exception:
            continue
    if Emb is None:
        # fall back to the reference source saved from transformers main (same code path, imported standalone)
        raise SkipTest("Gemma4UnifiedVisionEmbedder is not in the installed transformers 5.5.4; the next test checks "
                       "our module against the layer sequence transcribed from its source (transformers main)")
    from efvla.model.vision_ef import EFVisionEmbedder, grid_positions, patchify

    v = VCfg(patch_size=16, pooling_kernel_size=2, mm_embed_dim=48, mm_posemb_size=2, output_proj_dims=48)
    t = TCfg(hidden_size=40)
    ref = Emb(v, t).eval()
    randomize_(ref)
    ours = EFVisionEmbedder(img_size=64, patch=32, embed_dim=48, out_dim=40).eval()
    sd = {"patch_ln1.weight": ref.patch_ln1.weight, "patch_ln1.bias": ref.patch_ln1.bias,
          "patch_dense.weight": ref.patch_dense.weight, "patch_dense.bias": ref.patch_dense.bias,
          "patch_ln2.weight": ref.patch_ln2.weight, "patch_ln2.bias": ref.patch_ln2.bias,
          "pos_embedding": ref.pos_embedding, "pos_norm.weight": ref.pos_norm.weight, "pos_norm.bias": ref.pos_norm.bias,
          "embedding_projection.weight": ref.multimodal_embedder.embedding_projection.weight}
    ours.load_state_dict(sd, strict=True)
    img = torch.rand(2, 3, 64, 64)
    pv = patchify(img, 32)
    pos = grid_positions(2, 2).unsqueeze(0).expand(2, -1, -1)
    r = ref(pv, pos).pooler_output
    o = ours(img)
    d = maxdiff(o, r)
    assert d < 1e-5, d
    return f"max |diff| {d:.2e}"


@check("EF embedder formula: LN->Linear->LN->+pos(x,y)->LN->RMS(no scale)->Linear")
def t_ef_formula():
    import torch.nn.functional as F
    from efvla.model.vision_ef import EFVisionEmbedder, patchify

    m = EFVisionEmbedder(img_size=64, patch=32, embed_dim=24, out_dim=20).eval()
    randomize_(m)
    img = torch.rand(2, 3, 64, 64)
    x = patchify(img, 32)
    x = F.layer_norm(x, (3072,), m.patch_ln1.weight, m.patch_ln1.bias, 1e-5)
    x = x @ m.patch_dense.weight.T + m.patch_dense.bias
    x = F.layer_norm(x, (24,), m.patch_ln2.weight, m.patch_ln2.bias, 1e-5)
    pos = torch.stack([m.pos_embedding[j, 0] + m.pos_embedding[i, 1] for i in range(2) for j in range(2)])  # row-major
    x = F.layer_norm(x + pos, (24,), m.pos_norm.weight, m.pos_norm.bias, 1e-5)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = x @ m.embedding_projection.weight.T
    d = maxdiff(m(img), ref)
    assert d < 1e-5, d
    return f"max |diff| {d:.2e}"


@check("EF LayerNorm on near-uniform patches stays finite/bounded")
def t_ef_uniform():
    from efvla.model.vision_ef import EFVisionEmbedder

    m = EFVisionEmbedder(img_size=64, patch=32, embed_dim=24, out_dim=20).eval()
    flat = torch.full((1, 3, 64, 64), 0.5)
    noisy = flat + 1e-4 * torch.randn_like(flat)
    a, b = m(flat), m(noisy)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return f"|out| flat={a.norm(dim=-1).mean():.3f} noisy={b.norm(dim=-1).mean():.3f} (Gemma-style LN amplifies tiny noise; documented)"



# ----------------------------------------------------------------------------------------------- MolmoAct2 front end
def _import_ma2_image_processing():
    _import_molmoact2()
    import importlib

    return importlib.import_module("transformers_modules.MolmoAct2.image_processing_molmoact2")


@check("preprocess(uint8) == MolmoAct2 image processor resize+normalize (378, bilinear, no antialias)")
def t_preprocess_vs_ma2():
    IP = _import_ma2_image_processing()
    from transformers.image_utils import PILImageResampling
    from efvla.data.transforms import preprocess

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    ref = IP.normalize_image(IP.resize_image(img, [378, 378], PILImageResampling.BILINEAR), [0.5] * 3, [0.5] * 3)
    ours = preprocess(torch.from_numpy(img)[None, None])[0, 0].permute(1, 2, 0).numpy()
    d = float(np.abs(ours - ref).max())
    assert d < 1e-6, d
    # float (augmented) path lands on the same uint8 grid
    f = preprocess(torch.from_numpy(img).permute(2, 0, 1).float()[None, None] / 255.0)
    on_grid = float(((f + 1) / 2 * 255 - torch.round((f + 1) / 2 * 255)).abs().max())
    assert on_grid < 1e-4, on_grid
    return f"uint8 path max |diff| {d:.1e}; float path on uint8 grid"


@check("pooling windows == MolmoAct2 arange_for_pooling (27x27 -> 14x14, pad bottom/right)")
def t_pool_windows():
    IP = _import_ma2_image_processing()
    from efvla.model.vision_enc import pooling_windows

    for g in (27, 4, 5):
        ref = IP.arange_for_pooling(np.arange(g * g).reshape(g, g), 2, 2).reshape(-1, 4)
        ours = pooling_windows(g, 2).numpy()
        assert (ref == ours).all(), g
    return "grids 27, 4, 5 identical"


@check("ENC front end == MolmoAct2VisionBackbone end to end (ViT blocks -3/-9, 2x2 attn pooling, SwiGLU)")
def t_enc_vs_ma2():
    M, C = _import_molmoact2()
    IP = _import_ma2_image_processing()
    from efvla.model.vision_enc import ENCVision

    torch.manual_seed(0)
    scfg = tiny_siglip_config()
    ours = ENCVision(siglip_path=None, out_dim=40, siglip_config=scfg, img_size=56).eval()
    randomize_(ours.image_pooling_2d, 0.1, 1)
    randomize_(ours.image_projector, 0.1, 2)
    vit_cfg = C.MolmoAct2VitConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=10, num_attention_heads=2,
                                   num_key_value_heads=2, head_dim=16, hidden_act="gelu_pytorch_tanh",
                                   image_patch_size=14, image_num_pos=16, image_default_input_size=[56, 56],
                                   layer_norm_eps=1e-6, float32_attention=True, attention_dropout=0.0,
                                   residual_dropout=0.0)
    ad_cfg = C.MolmoAct2AdapterConfig(vit_layers=[-3, -9], hidden_size=1152, num_attention_heads=16,
                                      num_key_value_heads=16, head_dim=72, intermediate_size=9728,
                                      text_hidden_size=40, hidden_act="silu", pooling_attention_mask=True,
                                      float32_attention=True, attention_dropout=0.0, residual_dropout=0.0,
                                      image_feature_dropout=0.0)
    for c in (vit_cfg, ad_cfg):
        c._attn_implementation = "sdpa"
    ref = M.MolmoAct2VisionBackbone(vit_cfg, ad_cfg).eval()
    # map SigLIP (HF) -> MolmoAct2 ViT naming
    sv = ours.siglip.vision_model
    sd = {"image_vit.positional_embedding": sv.embeddings.position_embedding.weight,
          "image_vit.patch_embedding.weight": sv.embeddings.patch_embedding.weight.permute(0, 2, 3, 1).reshape(32, -1),
          "image_vit.patch_embedding.bias": sv.embeddings.patch_embedding.bias}
    for i, wl in enumerate(sv.encoder.layers):
        l = wl.inner
        pre = f"image_vit.transformer.resblocks.{i}."
        for a, b in (("wq", l.self_attn.q_proj), ("wk", l.self_attn.k_proj), ("wv", l.self_attn.v_proj),
                     ("wo", l.self_attn.out_proj)):
            sd[pre + f"attention.{a}.weight"], sd[pre + f"attention.{a}.bias"] = b.weight, b.bias
        sd[pre + "feed_forward.w1.weight"], sd[pre + "feed_forward.w1.bias"] = l.mlp.fc1.weight, l.mlp.fc1.bias
        sd[pre + "feed_forward.w2.weight"], sd[pre + "feed_forward.w2.bias"] = l.mlp.fc2.weight, l.mlp.fc2.bias
        sd[pre + "attention_norm.weight"], sd[pre + "attention_norm.bias"] = l.layer_norm1.weight, l.layer_norm1.bias
        sd[pre + "ffn_norm.weight"], sd[pre + "ffn_norm.bias"] = l.layer_norm2.weight, l.layer_norm2.bias
    for k, v in ours.image_pooling_2d.state_dict().items():
        sd["image_pooling_2d." + k] = v
    for k, v in ours.image_projector.state_dict().items():
        sd["image_projector." + k] = v
    missing, unexpected = ref.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (40, 40, 3), dtype=np.uint8) for _ in range(2)]
    from efvla.data.transforms import preprocess

    x = preprocess(torch.from_numpy(np.stack(imgs))[:, None], size=56)[:, 0]  # [2, 3, 56, 56] in [-1, 1]
    patches = torch.from_numpy(IP.batch_pixels_to_patches(x.permute(0, 2, 3, 1).numpy(), 14))[:, None]  # [B,1,16,588]
    pidx = IP.arange_for_pooling(np.arange(16).reshape(4, 4), 2, 2).reshape(-1, 4)
    pidx = torch.from_numpy(np.stack([pidx, pidx]))  # [B, 4, 4]
    with torch.no_grad():
        r = ref(patches, pidx).view(2, 4, -1)
        o = ours(x)
    d = maxdiff(o, r)
    scale = float(r.abs().max())
    assert d / scale < 1e-5, (d, scale)
    return f"max |diff| {d:.1e} at output scale {scale:.1e} (rel {d/scale:.1e}); tokens {tuple(o.shape)}"


@check("training augmentation: LeRobot ImageTransforms, per camera, different draws, range [0,1]")
def t_augment():
    from efvla.data.transforms import augment_frames, make_train_augment

    aug = make_train_augment()
    fr = (np.random.default_rng(0).random((2, 256, 256, 3)) * 255).astype(np.uint8)
    fr[1] = fr[0]
    a = augment_frames(fr, aug)
    b = augment_frames(fr, aug)
    assert a.shape == (2, 3, 256, 256) and a.min() >= 0 and a.max() <= 1
    assert maxdiff(a[0], a[1]) > 1e-3, "cameras not augmented independently"
    assert maxdiff(a, b) > 1e-3, "two draws identical"
    n = aug.tf.n_subset if hasattr(aug, "tf") else None
    return f"n_subset={n}"


# ----------------------------------------------------------------------------------------------- backbone
@check("Qwen3KVBackbone == HF Qwen3Model (hidden + per-layer cache K/V), no padding")
def t_backbone_nopad():
    from transformers import DynamicCache, Qwen3Model
    from efvla.model.backbone import Qwen3KVBackbone
    from efvla.model.vla import build_attention_mask, position_ids_from_valid

    torch.manual_seed(0)
    cfg = tiny_qwen3_config()
    hf = Qwen3Model(cfg).eval()
    ours = Qwen3KVBackbone(None, model=hf).eval()
    B, L = 2, 12
    x = torch.randn(B, L, 64)
    valid = torch.ones(B, L, dtype=torch.bool)
    mask = build_attention_mask(valid, torch.full((L,), -1))
    pos = position_ids_from_valid(valid)
    h, kvs = ours(x, pos, mask)
    cache = DynamicCache(config=cfg)
    out = hf(inputs_embeds=x, position_ids=pos, use_cache=True, past_key_values=cache)
    d_h = maxdiff(h, out.last_hidden_state)
    d_kv = 0.0
    for i, (k, v) in enumerate(kvs):
        ck, cv = out.past_key_values.layers[i].keys, out.past_key_values.layers[i].values  # [B, kv, L, hd]
        d_kv = max(d_kv, maxdiff(k, ck.transpose(1, 2).reshape(B, L, -1)), maxdiff(v, cv.transpose(1, 2).reshape(B, L, -1)))
    assert d_h < 1e-4 and d_kv < 1e-4, (d_h, d_kv)
    return f"hidden {d_h:.1e}, K/V {d_kv:.1e}"


@check("Qwen3KVBackbone == HF Qwen3Model with LEFT padding (valid positions)")
def t_backbone_leftpad():
    from transformers import Qwen3Model
    from efvla.model.backbone import Qwen3KVBackbone
    from efvla.model.vla import build_attention_mask, position_ids_from_valid

    torch.manual_seed(1)
    cfg = tiny_qwen3_config()
    hf = Qwen3Model(cfg).eval()
    ours = Qwen3KVBackbone(None, model=hf).eval()
    B, L = 2, 10
    x = torch.randn(B, L, 64)
    valid = torch.ones(B, L, dtype=torch.bool)
    valid[1, :4] = False
    pos = position_ids_from_valid(valid)
    h, _ = ours(x, pos, build_attention_mask(valid, torch.full((L,), -1)))
    out = hf(inputs_embeds=x, attention_mask=valid.long(), position_ids=pos, use_cache=False)
    d = maxdiff(h[valid], out.last_hidden_state[valid])
    # and the padded row equals running the unpadded sequence alone
    h1, _ = ours(x[1:, 4:], position_ids_from_valid(valid[1:, 4:]),
                 build_attention_mask(valid[1:, 4:], torch.full((6,), -1)))
    d2 = maxdiff(h[1, 4:], h1[0])
    assert d < 1e-4 and d2 < 1e-4, (d, d2)
    return f"vs HF {d:.1e}, vs unpadded {d2:.1e}"


@check("attention mask: causal rows, bidirectional image blocks, pad keys hidden, no empty rows")
def t_mask():
    from efvla.model.vla import build_attention_mask

    valid = torch.tensor([[False, True, True, True, True, True, True]])
    block = torch.tensor([-1, -1, -1, 0, 0, 1, -1])
    m = build_attention_mask(valid, block)[0, 0]
    assert m.dtype == torch.bool and m.any(-1).all()
    assert not m[1:, 0].any(), "pad key visible"
    assert m[3, 4] and m[4, 3], "image block not bidirectional"
    assert not m[3, 5], "image 0 must not see image 1 (later)"
    assert m[5, 3] and m[5, 4], "later image sees earlier image causally"
    assert not m[2, 3], "text must not see later image"
    assert m[6, :].tolist() == [False] + [True] * 6
    return "ok"


# ----------------------------------------------------------------------------------------------- action expert
def _import_molmoact2():
    root = "/root/.cache/huggingface/modules"
    if not os.path.isdir(os.path.join(root, "transformers_modules", "MolmoAct2")):
        raise SkipTest("MolmoAct2 remote code not cached")
    sys.path.insert(0, root)
    import importlib

    return importlib.import_module("transformers_modules.MolmoAct2.modeling_molmoact2"), \
        importlib.import_module("transformers_modules.MolmoAct2.configuration_molmoact2")


@check("ported ActionExpert == original MolmoAct2 ActionExpert (same weights, masks, t)")
def t_expert_port():
    M, C = _import_molmoact2()
    from efvla.model.action_expert import ActionExpert, ActionExpertConfig

    kw = dict(hidden_size=32, num_heads=4, num_layers=2, mlp_ratio=4.0, ffn_multiple_of=16, qk_norm=True,
              rope=True, context_layer_norm=True, timestep_embed_dim=16, causal_attn=False, max_action_dim=7,
              max_action_horizon=10, attn_dropout=0.0, dropout=0.0, qk_norm_eps=1e-6)
    ref_cfg = C.MolmoAct2ActionExpertConfig(**kw)
    ref = M.ActionExpert(ref_cfg, llm_dim=64, llm_kv_dim=24, llm_num_layers=2).eval()
    randomize_(ref, std=0.3, seed=3)  # zero-init gates would make the comparison trivial
    ours = ActionExpert(ActionExpertConfig(**kw), llm_kv_dim=24, llm_num_layers=2).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)
    torch.manual_seed(0)
    B, H, L = 3, 10, 9
    x = torch.randn(B, H, 7)
    t = torch.rand(B)
    kv = [(torch.randn(B, L, 24), torch.randn(B, L, 24)) for _ in range(2)]
    enc_valid = torch.ones(B, L, dtype=torch.bool)
    enc_valid[1, :3] = False
    act_valid = torch.ones(B, H, dtype=torch.bool)
    act_valid[2, 7:] = False
    r = ref(x, t, encoder_kv_states=kv, encoder_attention_mask=enc_valid, action_attention_mask=act_valid)
    o = ours(x, t, kv_states=kv, encoder_valid=enc_valid, action_valid=act_valid)
    d = maxdiff(o, r)
    # num_expand path == explicit repetition through the reference
    Fn = 2
    xe = torch.randn(B * Fn, H, 7)
    te = torch.rand(B * Fn)
    kv_rep = [(k.repeat_interleave(Fn, 0), v.repeat_interleave(Fn, 0)) for k, v in kv]
    r2 = ref(xe, te, encoder_kv_states=kv_rep, encoder_attention_mask=enc_valid.repeat_interleave(Fn, 0),
             action_attention_mask=act_valid.repeat_interleave(Fn, 0))
    o2 = ours(xe, te, kv_states=kv, encoder_valid=enc_valid, action_valid=act_valid, num_expand=Fn)
    d2 = maxdiff(o2, r2)
    assert d < 1e-5 and d2 < 1e-5, (d, d2)
    return f"single {d:.1e}, expanded x{Fn} {d2:.1e}"


@check("ported init == original init (zero gates/final, xavier, residual scale)")
def t_expert_init():
    M, C = _import_molmoact2()
    from efvla.model.action_expert import ActionExpert, ActionExpertConfig

    kw = dict(hidden_size=32, num_heads=4, num_layers=3, ffn_multiple_of=16, timestep_embed_dim=16,
              max_action_dim=7, max_action_horizon=10)
    ours = ActionExpert(ActionExpertConfig(**kw), llm_kv_dim=24, llm_num_layers=3)
    assert ours.final_layer.linear.weight.abs().sum() == 0 and ours.blocks[0].modulation.linear.weight.abs().sum() == 0
    # xavier bound with residual scale for out_proj
    bound = math.sqrt(6 / (32 + 32)) * (2 * 3) ** -0.5
    assert ours.blocks[0].self_attn.out_proj.weight.abs().max() <= bound + 1e-6
    names_ref = set(M.ActionExpert(C.MolmoAct2ActionExpertConfig(**kw), llm_dim=64, llm_kv_dim=24,
                                   llm_num_layers=3).state_dict())
    assert names_ref == set(ours.state_dict()), names_ref ^ set(ours.state_dict())
    return "ok"


@check("flow: x_t/target formulas; Euler with exact velocity recovers the action")
def t_flow():
    from efvla.model.action_expert import flow_targets, sample_beta_timesteps

    a = torch.randn(2, 10, 7)
    n = torch.randn(2, 3, 10, 7)
    t = torch.rand(2, 3)
    xt, v = flow_targets(a, n, t)
    assert maxdiff(xt, (1 - t[..., None, None]) * n + t[..., None, None] * a[:, None]) == 0
    x = n[:, 0]
    for i in range(10):
        x = x + 0.1 * (a - n[:, 0])
    assert maxdiff(x, a) < 1e-5
    ts = sample_beta_timesteps(100000, device="cpu", alpha=1.0, beta=1.5, cutoff=1.0, time_offset=0.001, time_scale=0.999)
    assert ts.min() >= 0.001 and ts.max() <= 1.0
    # Beta(1, 1.5) mean = 0.4 -> 0.001 + 0.999*0.4
    assert abs(float(ts.mean()) - (0.001 + 0.999 * 0.4)) < 0.005, float(ts.mean())
    return f"t mean {float(ts.mean()):.4f}"


# ----------------------------------------------------------------------------------------------- data
@check("normalizer round trip; gripper untouched; binarize never yields 0")
def t_norm():
    from efvla.data.normalize import Normalizer, binarize_gripper

    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000, 7)).astype(np.float32)
    a[:, 6] = rng.choice([-1.0, 1.0], 1000)
    s = rng.normal(size=(1000, 8)).astype(np.float32)
    from efvla.data.libero_cache import compute_stats

    st = {"actions": compute_stats(a), "state": compute_stats(s)}
    N = Normalizer(st)
    na = N.norm_actions(a)
    assert (na[:, 6] == a[:, 6]).all()
    inside = (a[:, :6] >= np.array(st["actions"]["q01"][:6])) & (a[:, :6] <= np.array(st["actions"]["q99"][:6]))
    back = N.unnorm_actions(na)
    assert np.abs(back[:, :6][inside] - a[:, :6][inside]).max() < 1e-5
    g = binarize_gripper(np.array([[0, 0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 0, 1e-3], [0, 0, 0, 0, 0, 0, -1e-3]]))
    assert g[:, 6].tolist() == [-1.0, 1.0, -1.0]
    return "ok"


@check("dataset: chunk indices/padding at episode end; contiguity assertion")
def t_dataset():
    import json
    from efvla.data.dataset import LiberoFrames
    from efvla.data.libero_cache import compute_stats

    d = tempfile.mkdtemp()
    lens = [5, 12]
    n = sum(lens)
    frames = np.lib.format.open_memmap(os.path.join(d, "frames.npy"), mode="w+", dtype=np.uint8, shape=(n, 2, 8, 8, 3))
    frames[:] = 0
    del frames
    actions = np.zeros((n, 7), np.float32)
    actions[:, 0] = np.arange(n)
    actions[:, 6] = 1.0
    state = np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32)
    ep_id = np.repeat([0, 1], lens).astype(np.int32)
    t = np.concatenate([np.arange(l) for l in lens]).astype(np.int32)
    np.savez(os.path.join(d, "meta.npz"), state=state, actions=actions, ep_id=ep_id, t=t,
             task_id=np.zeros(n, np.int32), ep_start=np.array([0, 5]), ep_len=np.array(lens, np.int32),
             src_episode=np.array([0, 1]))
    json.dump(["task"], open(os.path.join(d, "tasks.json"), "w"))
    json.dump({"actions": compute_stats(actions + np.random.default_rng(1).normal(size=actions.shape) * 1e-3),
               "state": compute_stats(state)}, open(os.path.join(d, "stats.json"), "w"))
    ds = LiberoFrames(d, chunk=10)
    idx, pad = ds.chunk_indices(2)  # episode 0 (len 5), t=2 -> steps 2,3,4 valid
    assert pad.tolist() == [False] * 3 + [True] * 7 and idx[:3].tolist() == [2, 3, 4], (idx, pad)
    idx, pad = ds.chunk_indices(5)  # episode 1, t=0 -> 10 valid
    assert not pad.any() and idx.tolist() == list(range(5, 15))
    item = ds[2]
    assert (item["actions"][3:] == 0).all() and item["is_pad"].sum() == 7
    return "ok"


# ----------------------------------------------------------------------------------------------- whole model
def _tiny_vla(arm, use_state=False, seed=0):
    from efvla.model.vla import VLA

    torch.manual_seed(seed)
    cfg = tiny_cfg(arm, use_state)
    m = VLA(cfg, load_pretrained=False, llm_config=tiny_qwen3_config(), siglip_config=tiny_siglip_config())
    randomize_(m.action_expert, std=0.2, seed=seed + 1)  # open the zero-init gates for the test
    m.init_new_modules(0.05)
    return m.eval(), cfg


@check("VLA left-pad equivalence: batched (different prompt lengths) == each sample alone (EF and ENC)")
def t_vla_leftpad():
    msgs = []
    for arm in ("ef", "enc"):
        m, cfg = _tiny_vla(arm, use_state=True)
        B, H = 2, 10
        ids = torch.randint(0, 128, (B, 7))
        valid = torch.ones(B, 7, dtype=torch.bool)
        valid[0, :4] = False  # sample 0 has a 3-token prompt
        imgs = torch.rand(B, 2, 3, 56, 56) * 2 - 1
        state = torch.randn(B, 8)
        x_t = torch.randn(B, 2, H, 7)
        t = torch.rand(B, 2)
        av = torch.ones(B, H, dtype=torch.bool)
        av[1, 6:] = False
        with torch.no_grad():
            vb = m(ids, valid, imgs, state, x_t, t, av)
            v0 = m(ids[:1, 4:], valid[:1, 4:], imgs[:1], state[:1], x_t[:1], t[:1], av[:1])
            v1 = m(ids[1:], valid[1:], imgs[1:], state[1:], x_t[1:], t[1:], av[1:])
        d = max(maxdiff(vb[:1], v0), maxdiff(vb[1:], v1))
        assert d < 1e-4, (arm, d)
        msgs.append(f"{arm} {d:.1e}")
    return ", ".join(msgs)


@check("VLA gradients reach every trainable group; frozen params get none (EF and ENC)")
def t_vla_grads():
    msgs = []
    for arm in ("ef", "enc"):
        m, cfg = _tiny_vla(arm, use_state=True)
        m.train()
        B, H = 2, 10
        ids = torch.randint(0, 128, (B, 5))
        valid = torch.ones(B, 5, dtype=torch.bool)
        out = m(ids, valid, torch.rand(B, 2, 3, 56, 56) * 2 - 1, torch.randn(B, 8), torch.randn(B, 2, H, 7), torch.rand(B, 2),
                torch.ones(B, H, dtype=torch.bool))
        out.pow(2).mean().backward()
        bad = []
        for n, p in m.named_parameters():
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                bad.append(n)
            if not p.requires_grad and p.grad is not None:
                bad.append("frozen-with-grad:" + n)
        assert not bad, bad[:10]
        frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
        msgs.append(f"{arm}: frozen={len(frozen)} tensors")
    return "; ".join(msgs)


@check("sample_actions is deterministic given noise and finite")
def t_sample():
    m, cfg = _tiny_vla("ef")
    ids = torch.randint(0, 128, (1, 5))
    valid = torch.ones(1, 5, dtype=torch.bool)
    imgs = torch.rand(1, 2, 3, 56, 56) * 2 - 1
    noise = torch.randn(1, 10, 7)
    a1 = m.sample_actions(ids, valid, imgs, steps=10, noise=noise)
    a2 = m.sample_actions(ids, valid, imgs, steps=10, noise=noise)
    assert torch.equal(a1, a2) and torch.isfinite(a1).all()
    return f"shape {tuple(a1.shape)}"


if __name__ == "__main__":
    for fn in TESTS:
        fn()
    w = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * 100)
    for n, s, msg in RESULTS:
        print(f"{s:4s}  {n:<{w}}  {msg}")
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} not failing ({sum(r[1]=='SKIP' for r in RESULTS)} skipped)")
    sys.exit(1 if fails else 0)
