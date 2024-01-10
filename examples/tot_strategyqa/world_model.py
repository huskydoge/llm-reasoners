from reasoners import WorldModel, LanguageModel

StrategyQAAction = str
StrategyQAExample = str
StrategyQAState = list[StrategyQAAction]

class StrategyQAWorldModel(WorldModel[StrategyQAState, StrategyQAAction, StrategyQAExample]):
    """
    strategyQA World Model
    
    State: [action_1, action_2, ...]
    Action: action
    """
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> StrategyQAState:
        return []
    
    def step(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[StrategyQAState, dict]:
        state = state.copy()
        
        # update state
        state.append(action)
        
        return state, {}
    
    def is_terminal(self, state: StrategyQAState) -> bool:
        # if "the answer is" in state[-1]:
        if len(state) > 0 and "the answer is" in state[-1].lower():
            return True
        
        return False
        