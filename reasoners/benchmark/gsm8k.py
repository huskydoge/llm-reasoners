import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random

class GSM8KEvaluator():
    def __init__(self, 
                 output_extractor,
                 answer_extractor,
                 init_prompt={},
                 disable_log=False,
                 disable_tqdm=False) -> None:
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self._dataset_name = 'gsm8k'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm

    def sample_l2m_prompt(self, shuffle_prompt=True, num_shot=4):
        prompt = {}
        if shuffle_prompt:
            decomp_examples = random.sample(self.init_prompt["decomposition_pool"], num_shot)
            solv_examples = random.sample(self.init_prompt["solving_pool"], num_shot)
        else:
            decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
            solv_examples = self.init_prompt["solving_pool"][:num_shot]
        prompt["decomposition"] = "".join(decomp_examples) + self.init_prompt["composition_prefix"]
        prompt["overall"] = "".join(decomp_examples) + self.init_prompt["overall_prefix"]
        prompt["solving"] = "".join(solv_examples) + self.init_prompt["solving_prefix"]
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = int(output)
            answer = int(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = float(output)
            answer = float(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer

    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 log_dir=None):

        self.dataset = datasets.load_dataset("gsm8k", "main", split=f"test[{resume}:]")
        
        if log_dir is None:
            log_dir = f'logs/{self._dataset_name}_'\
                      f'{reasoner.search_algo.__class__.__name__}/'\
                      f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

        correct_count = 0

        for i, example in enumerate(tqdm(self.dataset,
                                         total=resume + len(self.dataset),
                                         initial=resume,
                                         desc=self._dataset_name,
                                         disable=self.disable_tqdm)):
            
            algo_output = reasoner(self.input_processor(example),
                                   prompt=self.sample_l2m_prompt(
                                       shuffle_prompt=shuffle_prompt,
                                       num_shot=num_shot))
            
            output = self.output_extractor(algo_output)
            answer = self.answer_extractor(example)
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                      f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            if not self.disable_log:
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
            
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                    pickle.dump(algo_output, f)
        
        return accuracy