from reasoners import WorldModel, LanguageModel

GSM8KAction = str
GSM8KExample = str
GSM8KState = list[GSM8KAction]

class GSM8KWorldModel(WorldModel[GSM8KState, GSM8KAction, GSM8KExample]):
    """
    GSM8K World Model
    
    State: [action_1, action_2, ...]
    Action: action
    """
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> GSM8KState:
        return []
    
    def step(self, state: GSM8KState, action: GSM8KAction) -> tuple[GSM8KState, dict]:
        state = state.copy()
        
        # update state
        state.append(action)
        
        return state, {}
    
    def is_terminal(self, state: GSM8KState) -> bool:
        # if "the answer is" in state[-1]:
        if len(state) > 0 and "the answer is" in state[-1].lower():
            return True
        
        return False
        