#!/bin/bash
# Evaluate one checkpoint on all 10 tasks, tasks round-robin over the given GPUs (P processes per GPU):
#   bash scripts/eval_ckpt.sh <ckpt/model.pt> <gpus e.g. 0,1,2> [procs_per_gpu=2] [extra args e.g. --blank_images]
set -e
source $(dirname $0)/env.sh
CKPT=$1; GPUS=$2; shift 2
PPG=2
if [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+$ ]]; then PPG=$1; shift; fi
TAG=$(basename $(dirname $CKPT))
SUFFIX=""; [[ " $* " == *" --blank_images "* ]] && SUFFIX="_blank"
OUT=$(dirname $(dirname $CKPT))/eval_${TAG}${SUFFIX}
mkdir -p $OUT
cd $EFVLA
IFS=',' read -ra G <<< "$GPUS"
W=$(( ${#G[@]} * PPG ))
for ((w=0; w<W; w++)); do
  gpu=${G[$((w % ${#G[@]}))]}
  tasks=$(seq $w $W 9 | paste -sd,)
  [ -z "$tasks" ] && continue
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=0 $PY -m efvla.eval_libero --ckpt $CKPT --suite libero_object \
    --tasks $tasks --episodes 50 --out $OUT "$@" > $OUT/worker_$w.log 2>&1 &
done
wait
$PY -m efvla.aggregate $OUT | tee $OUT/summary.txt
