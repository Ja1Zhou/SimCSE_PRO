#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=6
eval_steps=75
python train.py \
    --model_name_or_path bert-base-uncased \
    --temp 0.05 \
    --pooler_type avg \
    --mlp_only_train \
    --train_file data/wiki1m_for_simcse.txt \
    --max_seq_length 32 \
    --output_dir result/nli-unsup-bert-avg \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps $eval_steps \
    --logging_steps $eval_steps \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
