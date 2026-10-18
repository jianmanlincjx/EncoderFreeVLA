"""Training entry (torchrun, FSDP2). Example:
  torchrun --nproc_per_node 3 -m efvla.train --arm ef --out_dir outputs/ef_object --steps 8000
"""
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader

from efvla.config import parse_config
from efvla.data.dataset import InfiniteRandomSampler, LiberoFrames, ensure_shm_copy
from efvla.data.transforms import preprocess
from efvla.model.action_expert import flow_targets, sample_beta_timesteps
from efvla.model.vla import VLA, Tokenize

PRETRAINED_PREFIXES = ("backbone.layers.", "backbone.norm.", "vision.siglip.")


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def lr_factor(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def masked_mse(pred, target, action_valid):
    """pred/target [B,F,H,A]; mean over valid (non-padded) chunk steps only (MolmoAct2 multiplies pads by 0
    and averages over all slots; we average over valid slots so short tails are not down-weighted)."""
    m = action_valid[:, None, :, None].to(torch.float32).expand_as(target)
    err = (pred.float() - target) ** 2
    return (err * m).sum() / m.sum().clamp(min=1.0)


def to_local(x):
    return x.full_tensor() if hasattr(x, "full_tensor") else x


def group_grad_norms(model):
    groups = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        key = ("backbone.layers" if n.startswith("backbone.layers.") else
               "backbone.norm" if n.startswith("backbone.norm.") else
               "vision.siglip" if n.startswith("vision.siglip.") else
               "vision.entry" if n.startswith("vision.") else
               "action_expert" if n.startswith("action_expert.") else n.split(".")[0])
        groups[key] = groups.get(key, 0.0) + float(g.float().pow(2).sum())
    keys = sorted(groups)
    t = torch.tensor([groups[k] for k in keys], device="cuda", dtype=torch.float64)
    dist.all_reduce(t)
    return {k: float(v) ** 0.5 for k, v in zip(keys, t.tolist())}


def main():
    cfg = parse_config()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    assert cfg.global_batch % (cfg.per_gpu_batch * world) == 0, (cfg.global_batch, cfg.per_gpu_batch, world)
    accum = cfg.global_batch // (cfg.per_gpu_batch * world)
    os.makedirs(cfg.out_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if rank == 0:
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            f.write(cfg.to_json())
    log(rank, f"[train] arm={cfg.arm} world={world} per_gpu={cfg.per_gpu_batch} accum={accum} "
              f"global={cfg.global_batch} steps={cfg.steps} out={cfg.out_dir} git={git_commit(repo)}")

    # ---------------- data
    if local_rank == 0:
        ensure_shm_copy(cfg.cache_dir, cfg.shm_dir)
    dist.barrier()
    data_dir = cfg.shm_dir or cfg.cache_dir
    ds = LiberoFrames(data_dir, cfg.chunk, augment=cfg.augment)
    tok = Tokenize(cfg.llm_path, cfg.prompt_template)
    loader = DataLoader(ds, batch_size=cfg.per_gpu_batch, sampler=InfiniteRandomSampler(len(ds), cfg.seed, rank),
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=cfg.num_workers > 0)
    it = iter(loader)
    log(rank, f"[data] {data_dir} frames={len(ds)} tasks={len(ds.tasks)}")

    # ---------------- model (fp32 master weights)
    model = VLA(cfg, load_pretrained=True)
    for n, p in model.named_parameters():
        assert p.dtype == torch.float32, (n, p.dtype)
    rms = model.text_embedding_rms()
    model.init_new_modules(rms)
    model.to(device)
    first = next(it)
    imgs0 = preprocess(first["images"].to(device), cfg.model_img_size)
    before, after = model.calibrate_vision_scale(imgs0, rms)
    log(rank, f"[init] text-emb RMS={rms:.4f}  vision-token RMS before={before:.4f} after={after:.4f}")
    if cfg.diag:
        with torch.no_grad():
            ids, val = tok.batch([ds.tasks[i] for i in first["task_id"].tolist()], device)
            x, valid, mask, pos = model.build_sequence(ids, val, imgs0, first["state"].to(device))
            Lt = ids.shape[1]
            seg = {"text": x[:, :Lt][val], "vision": x[:, Lt:Lt + cfg.n_cams * model.n_img_tokens].flatten(0, 1)}
            if cfg.use_state:
                seg["state"] = x[:, -1]
            log(rank, "[diag] token RMS " + json.dumps({k: round(float(v.float().pow(2).mean(-1).sqrt().mean()), 4)
                                                       for k, v in seg.items()}))
    model.backbone.set_grad_ckpt(cfg.grad_ckpt)
    if cfg.arm == "enc":
        model.vision.set_grad_ckpt(cfg.grad_ckpt)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vis = sum(p.numel() for p in model.vision.parameters())
    n_ae = sum(p.numel() for p in model.action_expert.parameters())
    log(rank, f"[params] total={n_total/1e9:.3f}B trainable={n_train/1e9:.3f}B vision={n_vis/1e6:.1f}M "
              f"action_expert={n_ae/1e6:.1f}M")

    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.backbone.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    if cfg.arm == "enc":
        for layer in model.vision.siglip.vision_model.encoder.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
    for block in model.action_expert.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    model.train()

    pre, new = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (pre if n.startswith(PRETRAINED_PREFIXES) else new).append(p)
    groups = [{"params": pre, "lr": cfg.lr_pretrained, "base_lr": cfg.lr_pretrained, "name": "pretrained"},
              {"params": new, "lr": cfg.lr_new, "base_lr": cfg.lr_new, "name": "new"}]
    log(rank, f"[optim] pretrained tensors={len(pre)} new tensors={len(new)}")
    optim = torch.optim.AdamW(groups, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
    params = [p for g in groups for p in g["params"]]

    stats_meta = {"stats": ds.stats, "tasks": ds.tasks}
    llm_cfg = model.backbone.config.to_dict()
    sig_cfg = model.vision.siglip.config.to_dict() if cfg.arm == "enc" else None

    def save(step):
        d = os.path.join(cfg.out_dir, f"ckpt_{step:06d}")
        os.makedirs(d, exist_ok=True)
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            tmp = os.path.join(d, f"model.pt.tmp{os.getpid()}")
            torch.save({"model": sd, "config": dataclasses.asdict(cfg), "llm_config": llm_cfg, "siglip_config": sig_cfg,
                        "step": step, "git": git_commit(repo), **stats_meta}, tmp)
            os.replace(tmp, os.path.join(d, "model.pt"))
        del sd
        if cfg.save_optimizer:
            import torch.distributed.checkpoint as dcp

            osd = get_optimizer_state_dict(model, optim)
            dcp.save({"optim": osd}, checkpoint_id=os.path.join(d, "optim"))
        dist.barrier()
        log(rank, f"[ckpt] saved {d}")

    def make_batch(b):
        images = preprocess(b["images"].to(device, non_blocking=True), cfg.model_img_size)
        ids, val = tok.batch([ds.tasks[i] for i in b["task_id"].tolist()], device)
        actions = b["actions"].to(device, non_blocking=True).float()
        action_valid = ~b["is_pad"].to(device, non_blocking=True)
        state = b["state"].to(device, non_blocking=True).float()
        return images, ids, val, state, actions, action_valid

    def flow_inputs(actions):
        B = actions.shape[0]
        Fn = cfg.num_flow_timesteps
        t = sample_beta_timesteps(B * Fn, device=device, alpha=cfg.flow_beta_alpha, beta=cfg.flow_beta_beta,
                                  cutoff=cfg.flow_cutoff, time_offset=cfg.flow_time_offset,
                                  time_scale=cfg.flow_time_scale).view(B, Fn).float()
        noise = torch.randn(B, Fn, *actions.shape[1:], device=device, dtype=torch.float32)
        x_t, target = flow_targets(actions, noise, t)
        return x_t, t, target

    fixed = first
    t_last = time.time()
    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    for step in range(cfg.steps):
        for g in optim.param_groups:
            ratio = cfg.min_lr / g["base_lr"]
            g["lr"] = g["base_lr"] * lr_factor(step, cfg.warmup, cfg.steps, ratio)
        loss_sum = 0.0
        for micro in range(accum):
            b = fixed if (cfg.overfit_batch or (step == 0 and micro == 0)) else next(it)
            images, ids, val, state, actions, action_valid = make_batch(b)
            x_t, t, target = flow_inputs(actions)
            pred = model(ids, val, images, state, x_t, t, action_valid)
            loss = masked_mse(pred, target, action_valid)
            (loss / accum).backward()
            loss_sum += float(loss.detach()) / accum
        if cfg.diag and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            # note: MolmoAct2's adaLN-zero init makes LLM/vision grads exactly 0 at step 0 (gates start at 0)
            gn = group_grad_norms(model)
            frozen = model.backbone.embed_tokens.weight.grad
            log(rank, f"[diag] step {step} grad norms " + json.dumps({k: float(f"{v:.3e}") for k, v in gn.items()}) +
                f"  embed_tokens.grad is None: {frozen is None}")
        gnorm = to_local(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        optim.step()
        optim.zero_grad(set_to_none=True)
        lt = torch.tensor([loss_sum], device=device)
        dist.all_reduce(lt)
        loss_mean = float(lt) / world
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            torch.cuda.synchronize()
            dt = (time.time() - t_last) / (cfg.log_every if step > 0 else 1)
            t_last = time.time()
            rec = {"step": step, "loss": round(loss_mean, 6), "grad_norm": round(float(gnorm), 4),
                   "lr_pre": optim.param_groups[0]["lr"], "lr_new": optim.param_groups[1]["lr"],
                   "s_per_step": round(dt, 3), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            log(rank, json.dumps(rec))
            if rank == 0:
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        if (step + 1) % cfg.save_every == 0 or step == cfg.steps - 1:
            save(step + 1)

    if cfg.diag:  # blank-image check on the fixed batch with identical noise/t
        model.eval()
        with torch.no_grad():
            images, ids, val, state, actions, action_valid = make_batch(fixed)
            torch.manual_seed(123)
            x_t, t, target = flow_inputs(actions)
            l_real = masked_mse(model(ids, val, images, state, x_t, t, action_valid), target, action_valid)
            l_blank = masked_mse(model(ids, val, torch.zeros_like(images), state, x_t, t, action_valid), target,
                                 action_valid)
            lr_ = torch.stack([l_real, l_blank]).clone()
            dist.all_reduce(lr_)
            log(rank, f"[diag] fixed-batch loss real={float(lr_[0])/world:.5f} blank-images={float(lr_[1])/world:.5f}")
    dist.barrier()
    log(rank, "[train] DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
