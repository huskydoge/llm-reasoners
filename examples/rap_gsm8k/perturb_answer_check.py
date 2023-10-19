import pickle
from datasets import Dataset
import utils
from reasoners.algorithm.beam_dfs import bDFSNode, bDFSResult
RESUME_B = 0
RESUME_E = 16
def increase_q(node: bDFSNode):
    if node.parent is None:
        node.q += 1
        return
    else:
        node.q += 1
        increase_q(node.parent)
data = Dataset.from_json('/data/haotian/RAP_tune/llm-reasoners/perturb_data.json')
def dfs_perturb_answer(node: bDFSNode, data_idx: int):##since the formmer answer is old answer to check the leakage
    flag = False
    node.q = 0
    if node.is_terminal:
        perturb_answer = utils.retrieve_answer_from_dataset_perturb(data['answer'][data_idx])
        output_answer = utils.retrieve_answer(node.state[-1].sub_question)
        correct = utils.judge_answer(output_answer, perturb_answer)
        if correct:
            increase_q(node) 
            flag = True
        return flag
    else:
        for child in node.children:
            flag |= dfs_perturb_answer(child, data_idx)
    
    return flag
correct_count = 0
for idx in range(RESUME_B, RESUME_E):
    bdfs_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/10172023-132150/algo_output/{idx+1}.pkl','rb'))
    root = bdfs_result.tree_state
    correct = dfs_perturb_answer(root, idx)
    correct_count += int(correct)
    accuracy = correct_count / (idx + 1)
    log_str = f'Case #{RESUME_B + idx + 1}: {correct=}; {accuracy=:.3f} ({correct_count}/{idx + 1})'
    with open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/perturbed/result.log','a') as f:
        print(log_str, file=f)
    with open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/perturbed/algo_output/{idx+1}.pkl', 'wb') as f:
        pickle.dump(bdfs_result, f)
