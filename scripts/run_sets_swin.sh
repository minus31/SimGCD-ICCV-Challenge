#!/bin/bash
set -e
set -x

bs=256
gradfromblock=11
epochs=200
supervised_weight=0.35
l_r=0.1

# #aircraft
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
    --dataset_name 'aircraft' \
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --epochs $epochs \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight $supervised_weight \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr $l_r \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name aircraft_simgcd > log/training/aircraft.txt

# #cub 
# CUDA_VISIBLE_DEVICES=1 python train_swin.py \
#     --dataset_name 'cub' \
#     --batch_size $bs \
#     --grad_from_block $gradfromblock \
#     --epochs $epochs \
#     --num_workers 8 \
#     --use_ssb_splits \
#     --sup_weight $supervised_weight \
#     --weight_decay 5e-5 \
#     --transform 'imagenet' \
#     --lr $l_r \
#     --eval_funcs 'v2' \
#     --warmup_teacher_temp 0.07 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 30 \
#     --memax_weight 2 \
#     --exp_name cub_simgcd  > log/training/cub.txt&


# # cars 
# CUDA_VISIBLE_DEVICES=2 python train_swin.py \
#     --dataset_name 'scars' \
#     --batch_size $bs \
#     --grad_from_block $gradfromblock \
#     --epochs $epochs \
#     --num_workers 8 \
#     --use_ssb_splits \
#     --sup_weight $supervised_weight \
#     --weight_decay 5e-5 \
#     --transform 'imagenet' \
#     --lr $l_r \
#     --eval_funcs 'v2' \
#     --warmup_teacher_temp 0.07 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 30 \
#     --memax_weight 1 \
#     --exp_name scars_simgcd > log/training/cars.txt &
