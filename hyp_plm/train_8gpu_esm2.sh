#!/usr/bin/env bash
# 八卡训练 ESM2（欧氏）。在 hyp_plm 目录下执行： ./train_8gpu_esm2.sh

set -e
cd "$(dirname "$0")"

# wandb：请先设置 API key，二选一：
# 1) 运行前执行: export WANDB_API_KEY='你的key'
# 2) 或取消下面一行注释并填入 key（勿提交到 git）
# export WANDB_API_KEY="wandb_v1_xxxxxxxx"

CONFIG=configs/train_8gpu_esm2.yaml
N_GPUS=8

torchrun --nproc_per_node="$N_GPUS" scripts/test_train.py --config "$CONFIG"
