import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float


StrategyQAState = list[SubResult]
StrategyQAAction = str
StrategyQAExample = str

class StrategyQAWorldModel(WorldModel[StrategyQAState, StrategyQAAction, StrategyQAExample]):
    """
    strategyQA World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 eos_token_id='\n',
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.eos_token_id = eos_token_id

    def init_state(self) -> list:
        return []

    def step(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[StrategyQAState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt['solving'] + "\n\n")
            f.write(f"Question 5: {self.example}\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(f"Question 5.{idx+1}: {q}\n")
                f.write(f"Answer 5.{idx+1}: {a}\n")
            f.write(f"Question 5.{len(state)+1}: {action}\n")
            f.write(f"Answer 5.{len(state)+1}:")
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        # print(f'====subanswer prompt====\n{model_input}\n====', flush=True)
        # print(f"====model input====\n{model_input}\n====", flush=True)

        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate([model_input] * num,
                                                    new_token_nums=256,
                                                    hide_input=True,
                                                    do_sample=True,
                                                    top_p=0.95,
                                                    temperature=self.temperature,
                                                   eos_token_id=self.eos_token_id).text
                for output in outputs:
                    result = output.strip().split('\n')[0]
                    print(f"subanswer output: {result}", flush=True)
                    answer = utils.retrieve_answer(result)
                    if answer is not None:
                        answer_dict[answer].append(result)

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(sorted_answer_dict[1][1]):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {'confidence': confidence}
        return state, aux

    def is_terminal(self, state: StrategyQAState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        elif len(state) > 0:
            ## try word match
            last_sub_words = set(state[-1].sub_question.lower().split(' '))
            overall_ques_words = set(self.example.lower().split(' '))
            new_words = last_sub_words - overall_ques_words
            if len(new_words) <= 1:
                return True
        return False
