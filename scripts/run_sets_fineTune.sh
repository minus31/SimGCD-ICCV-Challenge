#!/bin/bash
set -e
set -x

epochs=200
bs=256
gradfromblock=11

model_aircraft="/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/aircraft_simgcd_(13.09.2023_|_30.805)/checkpoints/model.pt"
model_cub="/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/cub_simgcd_(13.09.2023_|_46.897)/checkpoints/model.pt"
model_scars="/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/scars_simgcd_(13.09.2023_|_46.844)/checkpoints/model.pt"
sup_weight=0.35
l_r=0.03

#aircraft
CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name 'aircraft' \
    --warmup_model_dir $model_aircraft\
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --epochs $epochs \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight $sup_weight \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr $l_r \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name aircraft_simgcd > log/training/aircraft.txt&

#cub 
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name 'cub' \
    --warmup_model_dir $model_cub\
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --epochs $epochs \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight $sup_weight \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr $l_r \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --exp_name cub_simgcd  > log/training/cub.txt&

# cars 
CUDA_VISIBLE_DEVICES=3 python train.py \
    --dataset_name 'scars' \
    --warmup_model_dir $model_scars \
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --epochs $epochs \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight $sup_weight \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr $l_r \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name scars_simgcd > log/training/cars.txt&
