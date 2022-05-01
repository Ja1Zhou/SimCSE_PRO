#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client evaluation.py \
    --model_name_or_path bert-base-uncased \
    --pooler_type avg \
    --task_set sts \
    --mode test