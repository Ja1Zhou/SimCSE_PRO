export CUDA_VISIBLE_DEVICES=7
python evaluation.py \
    +data_args=prompt \
    +model_args=prompt_new_cl \
    +trainer_args=prompt \