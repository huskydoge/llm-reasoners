#!/bin/bash

prompt_dir='examples/tot_aqua/prompt.json'
log_dir='logs/tot-bfs-aqua'

mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=4
nohup python examples/tot_aqua/inference.py \
        --base_lm exllama --exllama_ckpt /data/yi --llama_size 70B --search_algo bfs \
        --prompt $prompt_dir --n_action 4 --beam_size 10 --temperature 0.8 --max_seq_len 3584 \
        --resume 0 --log_dir $log_dir \
        > $log_dir/log.txt 2> $log_dir/nohup.out &