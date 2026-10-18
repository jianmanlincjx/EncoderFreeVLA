# Encoder-Free VLA

Controlled comparison on a shared **text-only LLM backbone (Qwen3-4B)**, changing only the vision entry:

| arm | vision entry | params of entry |
|---|---|---|
| `ef`  | Gemma-4-style linear embedder (raw 32x32 patches, LN -> Linear -> LN -> +(x,y) pos -> LN -> RMSNorm -> Linear) | ~16M |
| `enc` | pretrained SigLIP2 so400m-patch16-256 + 2x2 merge + MLP connector | ~420M |

Everything else is identical: 64 tokens per camera (agentview + wrist), text-first sequence with bidirectional
attention inside each image, and a **MolmoAct2 action expert** (36 DiT blocks, width 768) that cross-attends to
every LLM layer's K/V and is trained with flow matching (Beta(1,1.5) times, 8 t per sample, 10-step Euler).
Optimisation follows starVLA's LIBERO recipe (pretrained 1e-5, new modules 1e-4, AdamW(0.9,0.95), cosine to 1e-6).

## Layout
```
efvla/config.py            all hyper-parameters (CLI overridable: --name value)
efvla/data/libero_cache.py build uint8 frame cache + stats from the openpi-format LIBERO parquet
efvla/data/dataset.py      frame dataset, 10-step chunks with padding mask
efvla/data/transforms.py   shared train/eval preprocessing (optional augmentation)
efvla/model/vision_ef.py   encoder-free embedder (port of Gemma4UnifiedVisionEmbedder)
efvla/model/vision_enc.py  SigLIP2 + merge + connector
efvla/model/backbone.py    Qwen3 layer loop returning per-layer post-RoPE K/V
efvla/model/action_expert.py  line-for-line port of MolmoAct2's ActionExpert + flow helpers
efvla/model/vla.py         root module: sequence, masks, training forward, sampling
efvla/train.py             torchrun + FSDP2 (fp32 master weights, bf16 compute)
efvla/eval_libero.py       closed-loop LIBERO eval (openpi protocol); efvla/aggregate.py sums results
tests/run_tests.py         equivalence tests vs HF Qwen3, original MolmoAct2 expert, Gemma embedder formula
scripts/check_data.py      orientation / state / gripper / language checks against the simulator
```

## Usage (fm1)
```bash
source scripts/env.sh
$PY -m efvla.data.libero_cache --suite libero_object --out data_cache/libero_object      # once, CPU
CUDA_VISIBLE_DEVICES= $PY -m tests.run_tests                                              # unit tests
bash scripts/smoke.sh ef 0,1,2 29611                                                       # smoke one arm
CUDA_VISIBLE_DEVICES=0,1,2 $TORCHRUN --nproc_per_node 3 --master_port 29611 -m efvla.train \
     --arm ef --out_dir outputs/ef_object --steps 8000 --global_batch 120 --per_gpu_batch 40
$PY -m efvla.eval_libero --ckpt outputs/ef_object/ckpt_008000/model.pt --tasks 0,1,2,3,4 --out outputs/ef_object/eval_8000
$PY -m efvla.aggregate outputs/ef_object/eval_8000
```

## Known factors
* The simulator (MuJoCo 3.8.1 in this venv) renders frames ~27% darker than the openpi dataset frames
  (mean RGB ~101 vs ~138); identical for both arms and for every baseline evaluated in this environment.
* LIBERO-Object is near saturation for strong baselines; use `--blank_images` evals and harder suites to separate arms.
