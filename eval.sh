#!/bin/bash
python evaluation.py \
    --model_name_or_path /home/zhejian/SimCSE/result/my-unsup-simcse-roberta-base \
    --pooler cls \
    --task_set sts \
    --mode test