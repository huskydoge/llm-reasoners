from reasoners import WorldModel, LanguageModel

AQuAAction = str
AQuAExample = str
AQuAState = list[AQuAAction]

class AQuAWorldModel(WorldModel[AQuAState, AQuAAction, AQuAExample]):
    """
    AQuA World Model
    
    State: [action_1, action_2, ...]
    Action: action
    """
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> AQuAState:
        return []
    
    def step(self, state: AQuAState, action: AQuAAction) -> tuple[AQuAState, dict]:
        state = state.copy()
        
        # update state
        state.append(action)
        
        return state, {}
    
    def is_terminal(self, state: AQuAState) -> bool:
        # if "the answer is" in state[-1]:
        if len(state) > 0 and "the answer is" in state[-1].lower():
            return True
        
        return False
        