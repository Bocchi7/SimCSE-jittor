#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path ../bert-base-chinese \
    --train_file SentEval/data/downstream/senteval_cn/PAWSX/PAWSX.train.data \
    --output_dir result/JittorFP16-unsup-simcse-pawsx-avg \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --dropout_rate 0.1 \
    --evaluation_strategy steps \
    --eval_chinese \
    --metric_for_best_model pawsx_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type avg \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
