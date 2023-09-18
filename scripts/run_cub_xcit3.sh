#!/bin/bash
set -e
set -x

#cub 
CUDA_VISIBLE_DEVICES=3 python train_xcit3.py \
    --dataset_name 'cub' \
    --warmup_model_dir '/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/cub_simgcd_(14.09.2023_|_32.878)/checkpoints/model.pt' \
    --batch_size 128 \
    --grad_from_block 23 \
    --only_representation_epochs 0 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.001 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --exp_name cub_simgcd  > log/training/cub.txt&


