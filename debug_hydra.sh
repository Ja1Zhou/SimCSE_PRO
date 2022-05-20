#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python -m debugpy --listen 127.0.0.1:5678 train.py \
    +model_args=self_weighted_bert_unsup_new_cl \
    +trainer_args=self_weighted_bert \