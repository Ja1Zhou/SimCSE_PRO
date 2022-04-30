#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    --model_name_or_path result/new-unsup-bert-avg \
    --pooler avg \
    --task_set full \
    --mode test