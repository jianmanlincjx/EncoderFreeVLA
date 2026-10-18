"""Experiment configuration. Every field can be overridden on the command line as --name value."""
import argparse
import dataclasses
import json
from dataclasses import dataclass, field

ROOT = "/pfs/pfs-Tr4Uts/JM_work"


@dataclass
class Config:
    # ---- arm ----
    arm: str = "ef"  # "ef" = linear embedder (encoder-free) | "enc" = pretrained SigLIP2 + connector

    # ---- paths ----
    llm_path: str = f"{ROOT}/hf_models/Qwen3-4B"
    siglip_path: str = f"{ROOT}/hf_models/siglip2-so400m-patch16-256"
    libero_root: str = "/pfs/pfs-Tr4Uts/libero"  # openpi-format LIBERO (LeRobot v2.1 parquet, images already rot180)
    suite: str = "libero_object"
    cache_dir: str = f"{ROOT}/encoder_free_vla/data_cache/libero_object"
    shm_dir: str = "/dev/shm/efvla_libero_object"  # training reads a copy from here (empty string = read cache_dir)
    out_dir: str = f"{ROOT}/encoder_free_vla/outputs/debug"

    # ---- inputs ----
    img_size: int = 256
    n_cams: int = 2  # agentview, wrist (fixed order)
    ef_patch: int = 32  # EF arm: 256/32 = 8x8 = 64 tokens per camera
    use_state: bool = False  # starVLA LIBERO recipe does not use proprio
    state_dim: int = 8
    augment: bool = False  # starVLA LIBERO recipe does not augment
    prompt_template: str = "Task: {instruction}. Action:"

    # ---- action expert (MolmoAct2 defaults) ----
    chunk: int = 10
    action_dim: int = 7
    ae_hidden: int = 768
    ae_heads: int = 8
    ae_mlp_ratio: float = 4.0
    ae_ffn_multiple_of: int = 256
    ae_timestep_dim: int = 256
    ae_qk_norm: bool = True
    ae_rope: bool = True
    ae_context_layer_norm: bool = True
    num_flow_timesteps: int = 8  # t samples per training example (share one LLM forward)
    flow_beta_alpha: float = 1.0
    flow_beta_beta: float = 1.5
    flow_time_offset: float = 0.001
    flow_time_scale: float = 0.999
    flow_cutoff: float = 1.0
    flow_infer_steps: int = 10

    # ---- optimisation (starVLA LIBERO recipe, scaled to one suite) ----
    steps: int = 8000
    global_batch: int = 120
    per_gpu_batch: int = 40
    warmup: int = 500
    lr_pretrained: float = 1e-5  # LLM layers / final norm / SigLIP
    lr_new: float = 1e-4  # randomly initialised: action expert, vision entry, camera emb, state proj
    min_lr: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_ckpt: bool = True
    seed: int = 42
    num_workers: int = 6
    log_every: int = 10
    save_every: int = 2000
    save_optimizer: bool = True

    # ---- smoke / debug ----
    overfit_batch: bool = False  # reuse the first batch forever
    diag: bool = False  # print grad norms per group, token RMS, blank-image loss delta

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False)


def _parse_bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {v}")


def parse_config(argv=None, base: Config | None = None) -> Config:
    base = base or Config()
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        default = getattr(base, f.name)
        t = type(default)
        if t is bool:
            p.add_argument(f"--{f.name}", type=_parse_bool, default=default)
        else:
            p.add_argument(f"--{f.name}", type=t, default=default)
    args = p.parse_args(argv)
    cfg = Config(**vars(args))
    assert cfg.arm in ("ef", "enc"), cfg.arm
    assert cfg.img_size % cfg.ef_patch == 0
    return cfg


def config_from_dict(d: dict) -> Config:
    names = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in names})
