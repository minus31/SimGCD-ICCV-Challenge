#!/bin/bash
set -e
set -x

bs=256
gradfromblock=10
epochs=200
supervised_weight=0.35
l_r=0.001


model_aircraft='/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/aircraft_simgcd_(07.09.2023_|_33.607)/checkpoints/model.pt'
model_cub='/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/cub_simgcd_(06.09.2023_|_55.754)/checkpoints/model.pt'
model_scars='/conor/SimGCD-ICCV-Challenge/dev_outputs/simgcd/log/scars_simgcd_(15.09.2023_|_39.335)/checkpoints/model.pt'

#cub 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'cub' \
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --warmup_model_dir $model_cub\
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
    --memax_weight 2 \
    --exp_name cub_simgcd  > log/training/cub.txt&

# #aircraft
CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name 'aircraft' \
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --warmup_model_dir $model_aircraft\
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
    --exp_name aircraft_simgcd > log/training/aircraft.txt&

# cars 
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset_name 'scars' \
    --batch_size $bs \
    --grad_from_block $gradfromblock \
    --warmup_model_dir $model_scars\
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
    --exp_name scars_simgcd > log/training/cars.txt&
