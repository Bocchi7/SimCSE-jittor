#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path ../gpt2  \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/jittorFP32-reproduce-unsup-simcse-gpt2-avg \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type avg \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --preprocessing_num_workers 32 \
    --do_train \
    --do_eval \
    "$@"
    # --fp16 \
