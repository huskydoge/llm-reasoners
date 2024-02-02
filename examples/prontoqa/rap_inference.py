import itertools
import os
import json
import fire

from dataset import ProntoQADataset
from reasoners import Reasoner

from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal

def rap_answer_extractor(mcts_result):
    if mcts_result.trace is None:
        return ""
    else:
        return "\n".join([mcts_result.trace[0][i].body for i in range(1, len(mcts_result.trace[0]) - 1)])
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(model_dir: str=  os.environ['LLAMA2_CKPTS'],
           mem_map: str = "[16, 22]",
           **search_algo_params):
    import numpy as np
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel(model_dir,
                                lora_dir=None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=mem_map,
                                log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    with open('examples/prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    world_model = ProntoQAWorldModel(base_model=language_model)
    search_config = ProntoQAConfig(base_model=language_model)
    search_algo = MCTS(output_trace_in_each_iter=True, cum_reward=np.mean, **search_algo_params)
    reasoner =  Reasoner(
            world_model=world_model,
            search_config=search_config,
            search_algo=search_algo
        )

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="rap",
        disable_log=False,
        output_extractor=rap_answer_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2]),
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/prontoqa/data/345hop_random_true.json'
        )
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4)
    print(f"accuracy: {accuracy}")


if __name__ == '__main__':
    fire.Fire(main)