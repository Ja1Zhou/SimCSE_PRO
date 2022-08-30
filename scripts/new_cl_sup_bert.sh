#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python train.py \
    +model_args=new_cl_sup \
    +training_args=default_sup \
    +data_args=default_sup