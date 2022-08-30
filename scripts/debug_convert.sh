#!/bin/bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client simcse_to_huggingface.py \
    --path /home/zhejian/SimCSE/result/reproduce-unsup-bert