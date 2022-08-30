#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python -m debugpy --listen 127.0.0.1:5678 evaluation.py \
    +data_args=prompt \
    +model_args=self_weighted_prompt_bert_unsup \
    +trainer_args=self_weighted_prompt_bert \