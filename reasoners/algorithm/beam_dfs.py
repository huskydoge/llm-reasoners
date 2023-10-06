import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
import utils
import numpy as np
from tqdm import trange
import random
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Trace

class bDFSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[bDFSNode]" = None, 
                is_terminal: bool = False):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(bDFSNode.id_iter)
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[bDFSNode]]' = []
        self._result = {}
        self.q = 0
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

class bDFSResult(NamedTuple):
    terminal_state: State
    tree_state: bDFSNode
    trace_in_each_iter: list[list[bDFSNode]] = None
    tree_state_after_each_iter: list[bDFSNode] = None

class bDFS(SearchAlgorithm, Generic[State,Action]):
    def __init__(self,
                 depth_limit: int = 5,
                 output_strategy: str = 'correct_or_none',
                 disable_tqdm: bool = True
                 ):
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.depth_limit = depth_limit
        self.output_strategy = output_strategy
        self.root: Optional[bDFSNode] = None
        self.disable_tqdm = disable_tqdm
        self.terminal_state = None

    def _construct_terminal_child(self, node: bDFSNode):
        children = []
        action = self.search_config.construct_final_action(node.state)
        child = bDFSNode(state=None, action=action, parent=node)
        children.append(child)
        node.children = children

    def _bdfs(self, node:bDFSNode):
        f1 = False
        if node.state is None and node.parent is not None:
            node.state = self.world_model.step(node.parent.state, node.action, self.correct_answer)
            node.is_terminal = self.world_model.is_terminal(node.state)
            # if aux.get('correct', True) and not node.is_terminal:
            #     self._construct_terminal_child(node)
            #     f1 = True
        if node.is_terminal:
            iter_output = utils.retrieve_answer(node.state)
            correct = utils.judge_answer(iter_output, self.correct_answer)
            if correct:
                self.terminal_state = node.state
                node.q = 1
                return 1
            else:
                node.q = 0
                return 0
        if not f1:
            children = []
            actions = self.search_config.get_actions(node.state)
            for action in actions:
                child = bDFSNode(state=None, action=action, parent=node)
                children.append(child)
            node.children.extend(children)
        for chd in node.children:
            node.q += self._bdfs(chd)
        return node.q


    def search(self):
        self.root = bDFSNode(state=self.world_model.init_state(), action=None, parent=None)
        self._bdfs(self.root)

    def __call__(self,
                 world_model: WorldModel[State, Action],
                 search_config: SearchConfig[State, Action],
                 correct_answer: str = None,
                 **kwargs) -> bDFSResult:
        bDFSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config
        self.correct_answer = correct_answer
        self.search()
        return bDFSResult(tree_state=self.root,tree_state_after_each_iter=None,terminal_state=self.terminal_state)