import pickle
# add path
import sys
sys.path.append('..')
from typing import Union
import os
# print(os.cwd())
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm import MCTSNode, bDFSNode

mcts_result = pickle.load(open('/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/09272023-191700/algo_output/2.pkl', 'rb'))
def gsm_node_data_factory(x: Union[MCTSNode, bDFSNode]) -> dict:
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

visualize(mcts_result, node_data_factory=gsm_node_data_factory)