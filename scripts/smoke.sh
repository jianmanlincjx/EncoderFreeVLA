#!/bin/bash
# Smoke test for one arm: bash scripts/smoke.sh <ef|enc> <gpus e.g. 0,1,2> <port>
# 1) 30 steps at the real batch (diag on, checkpoint + optimizer save at step 30)
# 2) 200-step overfit on one fixed batch (no augmentation)
# 3) checkpoint reload + 1 task x 2 episodes closed-loop eval
set -e
source $(dirname $0)/env.sh
ARM=$1; GPUS=$2; PORT=${3:-29611}
N=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT=$EFVLA/outputs/smoke_$ARM
mkdir -p $OUT
cd $EFVLA
echo "SMOKE_START $ARM gpus=$GPUS $(date)"
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node $N --master_port $PORT -m efvla.train \
  --arm $ARM --out_dir $OUT/real_batch --steps 30 --warmup 5 --per_gpu_batch 40 --global_batch $((40*N)) \
  --log_every 5 --save_every 30 --diag true 2>&1 | grep -vE "robosuite WARNING|private macro|setup_macros" | tee $OUT/real_batch.log
echo "SMOKE_REAL_DONE $ARM $(date)"
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node $N --master_port $PORT -m efvla.train \
  --arm $ARM --out_dir $OUT/overfit --steps 200 --warmup 20 --per_gpu_batch 8 --global_batch $((8*N)) \
  --overfit_batch true --augment false --log_every 20 --save_every 100000 --save_optimizer false --diag true 2>&1 \
  | grep -vE "robosuite WARNING|private macro|setup_macros" | tee $OUT/overfit.log
echo "SMOKE_OVERFIT_DONE $ARM $(date)"
G0=$(echo $GPUS | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$G0 MUJOCO_EGL_DEVICE_ID=0 $PY -m efvla.eval_libero --ckpt $OUT/real_batch/ckpt_000030/model.pt \
  --suite libero_object --tasks 0 --episodes 2 --video_every 1 --out $OUT/eval_smoke 2>&1 \
  | grep -vE "robosuite WARNING|private macro|setup_macros" | tee $OUT/eval_smoke.log
echo "SMOKE_ALL_DONE $ARM $(date)"
