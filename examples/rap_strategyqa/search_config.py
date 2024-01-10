import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import StrategyQAState, StrategyQAAction
from reasoners import SearchConfig, LanguageModel
import utils


class StrategyQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=4,
                 temperature=0.8,
                 eos_token_id='\n',
                 reward_alpha=0.25,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt = prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.subquestion_conf = {'Yes': 1.0, 'Maybe':0.5, 'No':0.1}

    def update_example(self, example: str, prompt: dict = None):
        super().update_example(example)
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example)[1]
            self.overall_question = self.example
            self.prompt = prompt


    def get_actions(self, state: StrategyQAState, ) -> list[StrategyQAAction]:
        with io.StringIO() as f:
            if len(state) == 0:
                f.write(self.prompt['decomposition'] + f'\n\nQ: {self.overall_question}\nA: To answer the question \"{self.overall_question}\", we need to know:')
            else:
                f.write(self.prompt["solving"] + "\n\n")
                f.write(f"Question 5: {self.example}\n")
                for idx, (q, a, _) in enumerate(state):
                    f.write(f"Question 5.{idx+1}: {q}\n")
                    f.write(f"Answer 5.{idx+1}: {a}\n")
                f.write(f"Question 5.{len(state)+1}:")
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" Now we can answer the question:")

            model_input = f.getvalue()
        
        # print(f'====model input====\n{model_input}\n====', flush=True)

        n_actions = self.n_actions

        # print(f"====model input====\n{model_input}\n====", flush=True)

        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                new_token_nums=256,
                                                hide_input=True,
                                                do_sample=True,
                                                top_k=32000,
                                                top_p=0.95,
                                                temperature=self.temperature,
                                                eos_token_id=self.eos_token_id).text
        

        outputs = [output.strip().split('\n')[0] for output in outputs]
        if len(state) == 0:
            for i, output in enumerate(outputs):
                subqs_list = utils.extract_subquestions(output[:-1])
                print('\n<<<< sub-questions list >>>>\n{}'.format(subqs_list), flush=True)
                q1 = subqs_list[0]
                if q1[0] != '"':
                    q1 = '"' + q1
                if q1[-1] != '"':
                    q1 = q1 + '"'
                outputs[i] = q1[1:-1]
        # print(f"====\nsub-question: {outputs}\n====")
        ### similar to is_terminal function in world
        if at_depth_limit:
            outputs = ["Now we can answer the question: " + self.overall_question]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if "Now we can answer the question:" in output:
                    outputs[i] = "Now we can answer the question: " + self.overall_question
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                last_sub_words = set(output.lower().split(' '))
                overall_ques_words = set(self.overall_question.lower().split(' '))
                new_words = last_sub_words - overall_ques_words
                if len(new_words) <= 1:
                    outputs[i] = "Now we can answer the question: " + self.overall_question

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def fast_reward(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.prompt["useful_prompt"])
            f.write(f"Question 7: {self.example}\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(f"Question 7.{idx+1}: {q}\n")
            f.write(f"Question 7.{len(state)+1}: {action}\n")
            f.write("Is the new question useful?")
            model_input = f.getvalue().replace('Now we can answer the question: ', '')

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)

        return fast_reward, {'r_useful': useful_prob}


    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: StrategyQAState, action: StrategyQAAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)
