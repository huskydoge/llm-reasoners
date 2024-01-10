import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import GSM8KState, GSM8KAction
from reasoners import SearchConfig, LanguageModel


class GSM8KConfig(SearchConfig):
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
    
    def get_actions(self, state: GSM8KState) -> list[GSM8KAction]:
        prompt = self.prompt["icl"]
        prompt += "\n\n"
        prompt += f"Question:\n{self.example}\nSteps:\n"
        for action in state:
            prompt += f"{action}\n"
        
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
        
        for output in outputs:
            print(f"output: {output}", flush=True)
        print("---------------------", flush=True)
        return outputs
    
    def fast_reward(self, state: GSM8KState, action: GSM8KAction) -> float:
        prompt = self.prompt["self-eval"]
        prompt += "\n\n"
        prompt += f"Question:\n{self.example}\nSteps:\n"
        for a in state:
            prompt += f"{a}\n"
        
        prompt += f"{action}\n"
        prompt += f"Is this step useful?\n"
        
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
    
    def reward(self, state: GSM8KState, action: GSM8KAction, **kwargs) -> float:
        # directly return the self_eval
        return kwargs["self_eval"], {"self_eval": kwargs["self_eval"]}