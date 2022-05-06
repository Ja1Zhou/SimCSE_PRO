#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    --model_name_or_path result/wiki-unique-unsup-bert-avg \
    --pooler_type avg \
    --task_set sts \
    --mode test