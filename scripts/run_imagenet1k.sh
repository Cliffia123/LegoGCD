#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_port=12348 --nproc_per_node=4 train_mp.py \
    --dataset_name 'imagenet_1k' \
    --batch_size 256 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --thr 0.7 \
    --exp_name imagenet1k_legogcd \
    --print_freq 100
