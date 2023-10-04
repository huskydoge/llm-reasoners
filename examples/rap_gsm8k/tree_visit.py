import json
import pandas as pd
from typing import Sequence, Union
import pickle
from reasoners.algorithm import MCTSNode, MCTSResult, BeamSearchNode, BeamSearchResult
from reasoners.algorithm.beam_dfs import bDFSNode, bDFSResult
from reasoners.visualization.tree_snapshot import NodeId, EdgeId, TreeSnapshot, NodeData, EdgeData
from datasets import load_dataset
import io
with open("/data/haotian/RAP_tune/llm-reasoners/examples/rap_gsm8k/prompts/DPO_example.json") as f:
    DPO_prompt = json.load(f)
with open("/data/haotian/RAP_tune/llm-reasoners/examples/rap_gsm8k/prompts/DPO_step_example.json") as f:
    DPO_step_prompt = json.load(f)
df = pd.DataFrame(columns=['id','question', 'response_j','response_k'])#j is better than k
df_final = pd.DataFrame(columns=['id','question', 'answer'])
gsm8k = load_dataset('gsm8k',"main",split='train')
def construct_subquestion(node: bDFSNode, example:str) -> str:
    with io.StringIO() as f:
        f.write(DPO_prompt["input"])
        f.write(DPO_prompt["question_prefix"] + example + "\n")
        for idx, (q, a, _) in enumerate(node.state):
            if idx == len(node.state) - 1:
                break
            f.write(DPO_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(DPO_prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
        f.write(DPO_prompt["subquestion_prefix"].format(len(node.state)))
        return f.getvalue()
def construct_steps(node: bDFSNode, example:str) -> str:
    with io.StringIO() as f:
        f.write(DPO_step_prompt["input"].format(example))
        for idx, (_, a, _) in enumerate(node.state):
            if idx == len(node.state) - 1:
                break
            else:
                f.write(DPO_step_prompt["step_prefix"].format(idx + 1) + " " + a.split("The answer is")[0].strip() + "\n")
        f.write(DPO_step_prompt["step_prefix"].format(len(node.state)) + " ")
        return f.getvalue()
def construct_final_steps(node: bDFSNode, example:str) -> str:
    with io.StringIO() as f:
        f.write(DPO_step_prompt["input"].format(example))
        question = f.getvalue()
    with io.StringIO() as f:
        for idx, (_, a, _) in enumerate(node.state):
            if idx == len(node.state) - 1:
                f.write(DPO_step_prompt["step_prefix"].format(idx + 1) + " " + a + "\n")
            else:
                f.write(DPO_step_prompt["step_prefix"].format(idx + 1) + " " + a.split("The answer is")[0].strip() + "\n")
        answer = f.getvalue()
    return question , answer
def construct_final_path(node: bDFSNode, example:str) -> str:
    with io.StringIO() as f:
        f.write(DPO_prompt["input"])
        f.write(DPO_prompt["question_prefix"] + example + "\n")
        question = f.getvalue()
    with io.StringIO() as f:
        for idx, (q, a, _) in enumerate(node.state):
            f.write(DPO_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(DPO_prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
        answer = f.getvalue()
    return question , answer
def dfs(x:bDFSNode, example,index):

    c_chs = []
    w_chs = []
    for c in x.children:
        if c.q > 0:
            c_chs.append(c)
        else:
            w_chs.append(c)
    for c_ch in c_chs:
        if len(w_chs) > 0:
            w_ch = w_chs.pop()
            # df.loc[len(df)] = [index, construct_subquestion(c_ch, example), c_ch.action, w_ch.action]
            # if c_ch.is_terminal:
            #     c_ch.action = c_ch.action + r'</s>'
            # if w_ch.is_terminal:
            #     w_ch.action = w_ch.action + r'</s>'
            df.loc[len(df)] = [index, construct_subquestion(c_ch, example), c_ch.action, w_ch.action]

    for c_ch in c_chs:
        dfs(c_ch, example, index)
def dfs_step(x:bDFSNode, example, index):
    c_chs = []
    w_chs = []
    for c in x.children:
        if c.q > 0:
            c_chs.append(c)
        else:
            w_chs.append(c)
    for c_ch in c_chs:
        if len(w_chs) > 0:
            w_ch = w_chs.pop()
            response_j = c_ch.state[-1].sub_answer
            response_k = w_ch.state[-1].sub_answer
            if not c_ch.is_terminal:
                response_j = response_j.split("The answer is")[0].strip()
            if not w_ch.is_terminal:
                response_k = response_k.split("The answer is")[0].strip()
            df.loc[len(df)] = [index, construct_steps(c_ch, example), response_j, response_k]
    for c_ch in c_chs:
        dfs_step(c_ch, example, index)
def dfs_final(x:bDFSNode, example, index):
    for c_ch in x.children:
        dfs_final(c_ch, example, index)
    if x.is_terminal:
        df_final.loc[len(df_final)] = [index, construct_final_path(x, example)[0], construct_final_path(x, example)[1]]
def dfs_step_final(x:bDFSNode, example, index):
    for c_ch in x.children:
        dfs_step_final(c_ch, example, index)
    if x.is_terminal:
        df_final.loc[len(df_final)] = [index, construct_final_steps(x, example)[0], construct_final_steps(x, example)[1]]
for i in range(1,1001):
    bdfs_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_bDFS/09272023-191700/algo_output/{i}.pkl', 'rb'))
    root = bdfs_result.tree_state
    # dfs(root, gsm8k[i-1]['question'], i-1)
    # dfs_final(root, gsm8k[i-1]['question'], i-1)
    dfs_step(root, gsm8k[i-1]['question'], i-1)
    # dfs_step_final(root, gsm8k[i-1]['question'], i-1)

# df_final.to_json('gsm8k_bDFS_SFT_STEP_1-1001.json', orient='records')
df.to_json('gsm8k_bDFS_step_1000.json', orient='records')