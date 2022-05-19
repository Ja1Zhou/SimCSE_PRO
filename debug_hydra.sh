#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python -m debugpy --listen 127.0.0.1:5678 train.py \
    +model_args=new_cl_sup \
    +training_args=default_sup \
    +data_args=default_sup