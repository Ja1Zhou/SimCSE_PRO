#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python evaluation.py \
    --model_name_or_path result/reproduce-unsup-bert-avg\
    --pooler avg \
    --task_set sts \
    --mode test