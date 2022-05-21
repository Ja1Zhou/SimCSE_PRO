#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    +model_args=self_weighted_bert_unsup \
    +trainer_args=self_weighted_bert \