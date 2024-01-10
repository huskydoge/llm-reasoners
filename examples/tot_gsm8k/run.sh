#!/bin/bash

log_dir='logs/tot_gsm8k_700'

mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=4,5
nohup python examples/tot_gsm8k/inference.py \
        --n_action 4 --n_confidence 8 --beam_size 10 --batch_size 2 \
        --resume 700 --log_dir $log_dir --exllama_mem_map 16,22 \
        >> $log_dir/output.log 2> $log_dir/nohup.out &