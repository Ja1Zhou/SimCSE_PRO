#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    +data_args=prompt \
    +model_args=self_weighted_prompt_bert_unsup \
    ++model_args.inference_prompt=2 \
    +trainer_args=self_weighted_prompt_bert \