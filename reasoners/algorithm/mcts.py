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


class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[MCTSNode]" = None, 
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
        self.id = next(MCTSNode.id_iter)
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = []
        self._result = {}
        self._result[1] = 0
        self._result[-1] = 0
        self._num_visits = 0
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def q(self) -> float:
        wins = self._result[1]
        loses = self._result[-1]
        return wins - loses
    @property
    def n(self) -> float:
        return self._num_visits + 1e-8


class MCTSResult(NamedTuple):
    terminal_state: State
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None


class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 20,
                 output_strategy: str = 'follow_max',
                 disable_tqdm: bool = True):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        assert output_strategy in ['follow_max', 'max_visit', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[MCTSNode] = None
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        self.flag = None
        self._correct_reward(path[-1])
        self._back_propagate(path)
        if self.output_strategy == 'last_iter':
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_iter = path
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._uct_select(node)

    def _uct(self, node: MCTSNode) -> float:
        return node.q/node.n + self.w_exp * np.sqrt(np.log(node.parent.n) / node.n)

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=self._uct)

    def _construct_terminal_child(self, node: MCTSNode):
        children = []
        action = self.search_config.construct_final_action(node.state)
        child = MCTSNode(state=None, action=action, parent=node)
        children.append(child)
        node.children = children


    def _expand(self, node: MCTSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action, self.correct_answer)
            node.is_terminal = self.world_model.is_terminal(node.state)
            if aux.get('correct', True):
                self._construct_terminal_child(node)
                #fix here
        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            child = MCTSNode(state=None, action=action, parent=node)
            children.append(child)
        node.children.extend(children)

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            node = random.choice(node.children)
            path.append(node)

    def _correct_reward(self, node: MCTSNode):
        if not node.is_terminal:
            self.flag = 0
            assert 1 == -1
            return
        else:
            iter_output = utils.retrieve_answer(node.state)
            correct = utils.judge_answer(iter_output, self.correct_answer)
            if correct:
                self.flag = 1
            else:
                self.flag = -1

    

    def _back_propagate(self, path: list[MCTSNode]):
        for node in reversed(path):
            node._num_visits += 1
            node._result[self.flag] += 1

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def search(self):
        self._output_iter = None
        self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.q)

    def __call__(self,
                 world_model: WorldModel[State, Action],
                 search_config: SearchConfig[State, Action],
                 correct_answer: str = None,
                 **kwargs) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config
        self.correct_answer = correct_answer
        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        return MCTSResult(terminal_state=terminal_state,
                          trace=trace,
                          trace_of_nodes=self._output_iter,
                          tree_state=self.root,
                          trace_in_each_iter=trace_in_each_iter,
                          tree_state_after_each_iter=tree_state_after_each_iter)


class MCTSAggregation(MCTS[State, Action], ABC):
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        assert weight_policy in ['edge', 'edge_inverse_depth']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode[State, Action]) -> Hashable:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list
        
        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])
