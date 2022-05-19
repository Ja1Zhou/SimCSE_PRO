#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python -m debugpy --listen 127.0.0.1:5678 train.py \
    +model_args=new_cl \
    ++data_args.train_file=data/wiki1m_test.txt
    ++training_args.eval_steps=1