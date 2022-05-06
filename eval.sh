#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    --model_name_or_path $1 \
    --pooler_type avg \
    --task_set sts \
    --mode test