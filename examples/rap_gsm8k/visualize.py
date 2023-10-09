from json import load
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
import fire
import datasets
from datasets import load_dataset
def main(idx: int):
    # mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/question_given/10042023-172629/algo_output/507.pkl', 'rb'))
    mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/10072023-163332/algo_output/{idx}.pkl', 'rb'))
    def gsm_node_data_factory(x: Union[MCTSNode, bDFSNode]) -> dict:
        if not x.state:
            return {}
        return {"question": x.state[-1].sub_question}
    print(idx)
    data = load_dataset('gsm8k', "main", split='train')
    print("The question is:" ,data['question'][idx-1])
    print("The answer is:" ,data['answer'][idx-1])
    visualize(mcts_result, node_data_factory=gsm_node_data_factory)

fire.Fire(main)