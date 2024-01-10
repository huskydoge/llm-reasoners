#!/bin/bash

prompt_dir='examples/tot_strategyqa/prompts/prompt.json'
log_dir='logs/dfs-strategyqa-test-70B'
data_file_path="examples/tot_strategyqa/data/test.json"

mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=1
nohup python examples/tot_strategyqa/inference_dfs.py \
        --base_lm exllama --exllama_ckpt /data/yi --llama_size 70B --reward_alpha 0.5 \
        --prompt $prompt_dir --n_action 4 --max_per_state 3 --total_states 100 --temperature 0.8 --max_seq_len 3584 \
        --resume 0 --log_dir $log_dir --data_file_path $data_file_path \
        > $log_dir/log.txt 2> $log_dir/nohup.out &