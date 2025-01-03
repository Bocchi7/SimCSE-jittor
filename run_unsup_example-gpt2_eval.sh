#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python evaluation.py \
    --model_name_or_path result/jittorFP16-unsup-simcse-gpt2-medium \
    --pooler avg \
    --task_set sts \
    --is_decoder \
    --mode test
    "$@"
    # --fp16 \
