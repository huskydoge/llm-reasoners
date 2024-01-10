from typing import Optional, Literal, Union

import re
from datetime import datetime
import json

from reasoners import LanguageModel, Reasoner
from reasoners.algorithm import BeamSearch, BeamSearchResult, DFS, DFSResult
from reasoners.benchmark import StrategyQAEvaluator

from world_model import StrategyQAWorldModel
from search_config import StrategyQAConfig

def retrieve_answer(output: Union[list, str, BeamSearchResult, DFSResult]) -> Optional[str]:
    
    
    if isinstance(output, BeamSearchResult):
        output = output.terminal_node.state
    if isinstance(output, DFSResult):
        output = output.terminal_state
    if isinstance(output, list):
        print(output, flush=True)
        output = output[-1]
        
    match = re.match(r'.*the answer is (.*)\.$', output)
    if match is None:
        ## negative word list
        if ' not ' in output or ' no ' in output or 'Not ' in output or 'No ' in output:
            answer = 'no'
        else:
            answer = ''
    else:
        answer = match[1]
    return answer

def tot_strategyQA(base_model: LanguageModel,
              prompt: dict,
              search_algo: str = 'bfs',
              resume: int = 0,
              n_action: int = 4,
              beam_size: int = 10,
              depth_limit: int = 7,
              temperature: float = 0.8,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              data_file_path: str = None,
              **search_algo_params):

    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/strategyQA_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    if search_algo == 'bfs':
        search_algo_params |= {'beam_size': beam_size, 'max_depth': depth_limit,
                            'early_terminate': True, 'reward_aggregator': 'last'}
        search_algo = BeamSearch(**search_algo_params)
    elif search_algo == 'dfs':
        search_algo_params |= {'max_per_state': 3, 'total_states': 10, 'depth': depth_limit}
        search_algo = DFS(**search_algo_params)

    world_model = StrategyQAWorldModel(base_model=base_model, prompt={})
    config = StrategyQAConfig(base_model=base_model, prompt={}, n_actions=n_action, temperature=temperature)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = StrategyQAEvaluator(
        output_extractor = retrieve_answer,
        init_prompt = prompt,
        disable_log = disable_log,
        disable_tqdm = disable_tqdm,
        data_file_path = data_file_path,
        sample_prompt_type="tot"
    )
    
    evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)



if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    from reasoners.lm import LlamaModel, LlamaCppModel, LlamaModel, ExLlamaModel
    import random
    import torch
    import torch.backends.cudnn

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    exllama_ckpt = os.environ.get("EXLLAMA_CKPT", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'exllama',
             llama_ckpt: str = llama_ckpts,
             llama_2_ckpt: str = llama_2_ckpts,
             exllama_ckpt: str = exllama_ckpt,
             llama_size: str = '30B',
             mem_map: list[int] = None,
             llama_cpp_path: str = None,
             batch_size: int = 2,
             max_seq_len: int = 2048,
             prompt: str = 'examples/rap_strategyQA/prompts/prompt.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
        # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

        with open(prompt, 'r') as f:
            prompt = json.load(f)
        if base_lm == 'llama':
            base_model = LlamaModel(llama_ckpt, llama_size, max_batch_size=batch_size, max_seq_len=max_seq_len)
        elif base_lm == 'llama.cpp':
            base_model = LlamaCppModel(llama_cpp_path)
        elif base_lm == 'llama2':
            base_model = LlamaModel(llama_2_ckpt, llama_size, max_batch_size=batch_size)
        elif base_lm == 'exllama':
            device = torch.device("cuda:0")
            base_model = ExLlamaModel(
                model_dir = f"{exllama_ckpt}/Llama-2-{llama_size}-GPTQ",
                lora_dir = None,
                device = device,
                max_batch_size = batch_size,
                max_new_tokens = 256,
                max_seq_length = max_seq_len,
                mem_map = mem_map
            )
                
        else:
            assert False, f'cannot resolve {base_lm=}'
        tot_strategyQA(base_model=base_model,
                    prompt=prompt,
                    disable_log=disable_log or local_rank != 0,
                    disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)
