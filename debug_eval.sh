#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client evaluation.py \
    --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased\
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test