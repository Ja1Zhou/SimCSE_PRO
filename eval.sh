#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python evaluation.py \
    --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased\
    --pooler cls \
    --task_set sts \
    --mode test