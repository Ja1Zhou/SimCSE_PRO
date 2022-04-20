#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python evaluation.py \
    --model_name_or_path result/reproduce-unsup-bert-cls\
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test