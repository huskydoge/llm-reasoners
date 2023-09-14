import pickle
# add path
import sys
sys.path.append('..')
import os
# print(os.cwd())
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
mcts_result = pickle.load(open('/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_MCTS/09132023-062200/algo_output/1.pkl', 'rb'))
def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
visualize(mcts_result, node_data_factory=gsm_node_data_factory)