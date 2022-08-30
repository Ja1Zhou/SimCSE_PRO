#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python train.py \
    +model_args=self_weighted_bert_unsup_new_cl \
    +trainer_args=self_weighted_bert \