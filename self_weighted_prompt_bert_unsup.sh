#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python train.py \
    +data_args=prompt \
    +model_args=self_weighted_prompt_bert_unsup \
    +trainer_args=self_weighted_prompt_bert \