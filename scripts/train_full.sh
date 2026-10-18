#!/bin/bash
# Full LIBERO-Object run for one arm (starVLA recipe scaled to one suite):
#   bash scripts/train_full.sh <ef|enc> <gpus e.g. 0,1,2> <port> [extra args...]
# 3 GPUs x 40 = global batch 120, 8000 steps (~14 epochs of 66,984 frames), warmup 500, ckpt every 2000.
set -e
source $(dirname $0)/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM=$1; GPUS=$2; PORT=$3; shift 3
N=$(echo $GPUS | tr ',' '\n' | wc -l)
PER=$((120 / N))
OUT=$EFVLA/outputs/${ARM}_object
mkdir -p $OUT
cd $EFVLA
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node $N --master_port $PORT -m efvla.train \
  --arm $ARM --out_dir $OUT --steps 8000 --warmup 500 --global_batch 120 --per_gpu_batch $PER \
  --save_every 2000 --log_every 20 "$@" 2>&1 | grep --line-buffered -vE "robosuite WARNING|private macro|setup_macros|NCCL INFO" \
  > $OUT/train.log
