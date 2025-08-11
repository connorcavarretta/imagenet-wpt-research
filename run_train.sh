#!/usr/bin/env bash
set -euo pipefail

# GPUs / data root
export CUDA_VISIBLE_DEVICES=0,1,2
export IMAGENET_PATH=/home/hongshen/work/data/ILSVRC/Data/CLS-LOC
export PYTHONUNBUFFERED=1

# NCCL: P2P caused hangs on this machine
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_COLLNET_ENABLE=0
# If socket selection ever causes issues, uncomment:
# export NCCL_SOCKET_IFNAME=eno2np1

# File descriptors (lots of files on ImageNet)
ulimit -n 65536 || true

# Run from repo root
cd /home/hongshen/work/imagenet-wpt-research

# Train (per‑GPU batch × GPUs = global batch; LR auto‑scales in train.py)
torchrun --standalone --nproc_per_node=3 \
  train.py \
  --batch-size 256 \
  --num-workers 8 \
  --epochs 300 \
  --warmup-epochs 20
