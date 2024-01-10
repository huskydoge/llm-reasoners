import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import AQuAState, AQuAAction
from reasoners import SearchConfig, LanguageModel


class AQuAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 temperature=0.8) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt = prompt
        self.temperature = temperature
        self.n_actions = n_actions
    
    def get_actions(self, state: AQuAState) -> list[AQuAAction]:
        prompt = self.prompt["icl"]
        prompt += "\n\n"
        prompt += f"Question:\n{self.example}\nAnswer:\n"
        for action in state:
            prompt += f"{action}\n"
        
        # print(f"Prompt: {prompt}", flush=True)
        
        ouputs = self.base_model.generate([prompt],
                                          num_return_sequences=self.n_actions,
                                          max_length=128,
                                          eos_token_id="\n",
                                          temperature=self.temperature,
                                          do_sample=True,
                                          hide_input=True).text
        outputs = [output.split("\n")[0] for output in ouputs]
        # deduplicate
        outputs = list(dict.fromkeys(outputs))
        # remove empty or only whitespace or only \n outputs
        outputs = [output for output in outputs if not re.match(r'^\s*$', output)]
        
        for output in outputs:
            print(f"output: {output}", flush=True)
        print("---------------------", flush=True)
        return outputs
    
    def fast_reward(self, state: AQuAState, action: AQuAAction) -> float:
        prompt = self.prompt["self-eval"]
        prompt += "\n\n"
        prompt += f"Question:\n{self.example}\nAnswer:\n"
        for a in state:
            prompt += f"{a}\n"
        
        prompt += f"{action}\n"
        prompt += f"Is this step useful?\n"
        
        # print(f"Prompt: {prompt}", flush=True)
        
        self_eval = self.base_model.get_loglikelihood(prompt, [prompt + "Yes"])[0]
        
        
        print(f"Action: {action}", flush=True)
        print(f"Self-eval: {self_eval}", flush=True)
        print("---------------------", flush=True)
        
        # # for safety, let's prompt with this prompt
        # output = self.base_model.generate([prompt],
        #                                   num_return_sequences=1,
        #                                   max_length=128,
        #                                   eos_token_id="\n",
        #                                   temperature=0,
        #                                   do_sample=True,
        #                                   hide_input=True).text[0]
        
        # print(f"output: {output}", flush=True)
        # print("---------------------", flush=True)
        
        return self_eval, {"self_eval": self_eval}
    
    def reward(self, state: AQuAState, action: AQuAAction, **kwargs) -> float:
        # directly return the self_eval
        return kwargs["self_eval"], {"self_eval": kwargs["self_eval"]}