import pickle
from typing import Type, Callable, Optional, Literal

import numpy as np
from reasoners.visualization import TreeLog
from tqdm import tqdm
from datetime import datetime
import json

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, BeamSearch, BeamSearchResult
from reasoners.benchmark import StrategyQAEvaluator

from world_model import StrategyQAWorldModel, StrategyQAState, StrategyQAAction
from search_config import StrategyQAConfig
from utils import retrieve_answer



def node_visualizer(x: MCTSNode[StrategyQAState, StrategyQAAction]):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

def rap_cum_reward(cum_rewards):
    return sum(cum_rewards) / (len(cum_rewards) + 1)

def rap_strategyQA(base_model: LanguageModel,
              prompt: dict,
              search_algo: Type[SearchAlgorithm] = BeamSearch,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              beam_size: int = 10,
              depth_limit: int = 7,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 1,
              eos_token_id='\n',
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

    search_algo_params |= {'beam_size': beam_size, 'max_depth': depth_limit,
                            'early_terminate': True, 'reward_aggregator': 'last'}

    eos_token_id = [13] # '\n', we do not want to use 29871
    world_model = StrategyQAWorldModel(base_model=base_model, prompt={}, eos_token_id=eos_token_id,
                                n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = StrategyQAConfig(base_model=base_model, prompt={}, eos_token_id=eos_token_id,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = StrategyQAEvaluator(
        output_extractor = retrieve_answer,
        init_prompt = prompt,
        disable_log = disable_log,
        disable_tqdm = disable_tqdm,
        data_file_path = data_file_path
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
             reward_alpha: float = 0.5,
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
        rap_strategyQA(base_model=base_model,
                    prompt=prompt,
                    batch_size=batch_size,
                    reward_alpha=reward_alpha,
                    disable_log=disable_log or local_rank != 0,
                    disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)
