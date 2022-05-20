#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python -m debugpy --listen 127.0.0.1:5678 train.py \
    +data_args=prompt \
    +model_args=prompt_new_cl \
    +trainer_args=prompt \