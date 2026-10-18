"""Reload a checkpoint into a plain (non-FSDP) model and measure the flow loss on real data.
A correctly restored model reproduces the end-of-training loss; a broken reload falls back to ~init loss.
Also reports the loss with blanked images (vision reliance) and checks sampling determinism.
  python scripts/check_ckpt.py --ckpt outputs/x/ckpt_000030/model.pt [--batches 4 --bs 16]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efvla.data.dataset import LiberoFrames  # noqa: E402
from efvla.data.transforms import preprocess  # noqa: E402
from efvla.model.action_expert import flow_targets, sample_beta_timesteps  # noqa: E402
from efvla.model.vla import Tokenize  # noqa: E402
from efvla.policy import load_model  # noqa: E402
from efvla.train import masked_mse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bf16")
    a = ap.parse_args()
    dev = torch.device("cuda")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model, cfg, ck = load_model(a.ckpt, dev, dtype)
    ds = LiberoFrames(cfg.shm_dir if os.path.exists(cfg.shm_dir) else cfg.cache_dir, cfg.chunk, augment=False)
    tok = Tokenize(cfg.llm_path, cfg.prompt_template)
    rng = np.random.default_rng(0)
    g = torch.Generator(device=dev).manual_seed(0)
    real, blank = [], []
    for _ in range(a.batches):
        idx = rng.choice(len(ds), a.bs, replace=False)
        items = [ds[i] for i in idx]
        imgs = preprocess(torch.stack([x["images"] for x in items]).to(dev), cfg.model_img_size)
        acts = torch.stack([x["actions"] for x in items]).to(dev).float()
        av = ~torch.stack([x["is_pad"] for x in items]).to(dev)
        st = torch.stack([x["state"] for x in items]).to(dev)
        ids, val = tok.batch([ds.tasks[x["task_id"]] for x in items], dev)
        Fn = cfg.num_flow_timesteps
        t = sample_beta_timesteps(a.bs * Fn, device=dev, alpha=cfg.flow_beta_alpha, beta=cfg.flow_beta_beta,
                                  cutoff=cfg.flow_cutoff, time_offset=cfg.flow_time_offset,
                                  time_scale=cfg.flow_time_scale).view(a.bs, Fn).float()
        noise = torch.randn(a.bs, Fn, cfg.chunk, cfg.action_dim, device=dev, generator=g)
        x_t, target = flow_targets(acts, noise, t)
        with torch.no_grad():
            p = model(ids, val, imgs.to(dtype), st.to(dtype), x_t.to(dtype), t.to(dtype), av)
            pb = model(ids, val, torch.zeros_like(imgs).to(dtype), st.to(dtype), x_t.to(dtype), t.to(dtype), av)
        real.append(float(masked_mse(p, target, av)))
        blank.append(float(masked_mse(pb, target, av)))
    n = torch.randn(1, cfg.chunk, cfg.action_dim, device=dev, generator=g)
    s1 = model.sample_actions(ids[:1], val[:1], imgs[:1].to(dtype), st[:1].to(dtype), steps=cfg.flow_infer_steps, noise=n)
    s2 = model.sample_actions(ids[:1], val[:1], imgs[:1].to(dtype), st[:1].to(dtype), steps=cfg.flow_infer_steps, noise=n)
    print(f"[ckpt] step={ck['step']} arm={cfg.arm} loss real={np.mean(real):.4f} blank-images={np.mean(blank):.4f} "
          f"sample deterministic={bool(torch.equal(s1, s2))} finite={bool(torch.isfinite(s1).all())}")


if __name__ == "__main__":
    main()
