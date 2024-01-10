import datasets
import json
from tqdm import tqdm
import torch
import os, pickle, re
from datetime import datetime
import sys
import random
from reasoners import Evaluator
from reasoners.visualization import TreeLog
from reasoners.algorithm import MCTS
from collections import Counter

def node_visualizer(x):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


class StrategyQAEvaluator(Evaluator):
    def __init__(self, 
                 output_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="rap",
                 data_file_path=None,
                 ) -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = lambda x: x["answer"]
        self.input_processor = lambda x: x["question"]
        if data_file_path is None:
            self.full_dataset = datasets.load_dataset('amydeng2000/strategy-qa', split='test')
        else:
            with open(data_file_path) as f:
                self.full_dataset = json.load(f)

        self._dataset_name = "strategy-qa"
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4,
                      sample_prompt_type="rap"):

        if sample_prompt_type == "rap":
            prompt = {}

            if shuffle_prompt:
                decomp_examples = random.sample(self.init_prompt["decomposition_pool"], num_shot)
                solv_examples = random.sample(self.init_prompt["solving_pool"], num_shot)
            else:
                decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
                solv_examples = self.init_prompt["solving_pool"][:num_shot]
            
            prompt["decomposition"] = "\n\n".join(decomp_examples)
            prompt["solving"] = "\n\n".join([ex.replace("{INDEX}", str(i+1)) for i, ex in enumerate(solv_examples)])
            if "useful_prompt" in self.init_prompt:
                prompt["useful_prompt"] = self.init_prompt["useful_prompt"]
        
        elif sample_prompt_type == "tot":
            prompt = {}
            
            if shuffle_prompt:
                cot_prompt = random.sample(self.init_prompt["solving_pool"], num_shot)
            else:
                cot_prompt = self.init_prompt["solving_pool"][:num_shot]
            
            prompt["icl"] = "\n\n".join(cot_prompt)
            
            if "self-eval" in self.init_prompt:
                prompt["self-eval"] = self.init_prompt["self-eval"]
        
        return prompt

        
    def eval_output(self, answer, output):
        if output is None:
            return False
        
        # False vs no and True vs yes
        answer = "no" if not answer else "yes"
        
        return answer == output.strip().lower()