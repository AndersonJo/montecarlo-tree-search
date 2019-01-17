from copy import deepcopy
from math import sqrt
from random import random, choice
from typing import Union, List, Tuple

import numpy as np
from tqdm import tqdm

from mcts.env import BaseEnv


class Node:
    def __init__(self, action, value: float = 0., visit: int = 1):
        self.action = action
        self.value = value
        self.n_visit = visit
        self.children = dict()

    def has_child(self) -> bool:
        return bool(self.children)

    def link_next_state(self, state):
        self.children.setdefault(state, 0)

    def __repr__(self):
        return '<Node.{0} w:{1} v:{2}>'.format(self.action, self.value, self.n_visit)


class MCTS(object):
    def __init__(self, env: BaseEnv, simulation: int = 10, c: float = sqrt(2), exploration_rate: float = 0.5,
                 max_depth: int = 50, max_depth_penalty: float = -1):
        self.env = env
        self.simulation = simulation
        self.history = []
        self.nodes = dict()

        self.max_depth = max_depth
        self.max_depth_penalty = max_depth_penalty  # Tree reached the maximum depth, it will get penalty
        self.exploration_rate = exploration_rate
        self.C = c  # Exploration constant

    @property
    def cur_node(self):
        if self.history:
            return self.nodes[self.history[-1]]

        return None

    @property
    def cur_depth(self):
        return len(self.history)

    def get_action(self, node: Node, legal_actions) -> Union[None, Node]:
        """
        :return: next node's action
        """
        if self.cur_depth >= self.max_depth:
            return None

        if not node.has_child():  # reached the leaf node
            action = self.rand_action(node, legal_actions)
        elif random() <= self.exploration_rate and len(legal_actions) >= 1:
            action = self.rand_action(node, legal_actions)
        else:
            next_node, uct = self.calculate_uct(node)
            action = next_node.action

        return action

    def rand_action(self, node, legal_actions) -> Node:
        assert len(legal_actions) >= 1

        tried_actions = {self.nodes[s] for s in node.children}
        action = choice(list(set(legal_actions) - tried_actions))

        # new_node = Node(action=rand_action)
        # node.link_next_state(state)

        return action

    def calculate_uct(self, node) -> Tuple[Node, float]:

        nodes = [self.nodes[key] for key in node.children]
        ucts = map(lambda c: (c, (c.value / c.n_visit) + self.C * np.sqrt(np.log(node.n_visit) / c.n_visit)),
                   nodes)
        return sorted(ucts, key=lambda c: -c[1])[0]

    def expand(self, cur_node: Union[Node, None], state, action):
        new_node = self.nodes.get(state, None)

        if new_node is None:
            new_node = Node(action=action)
            self.nodes[state] = new_node

        if cur_node is not None:
            cur_node.link_next_state(state)

        self.history.append(state)
        return

    def simulate(self, env, state):
        env = deepcopy(env)
        nodes = deepcopy(self.nodes)

        for i in range(self.simulation):
            self._simulate(env, state, nodes)

    def _simulate(self, env, state, nodes):
        env.render()
        import ipdb
        ipdb.set_trace()

    def train(self, epochs: int = 10000, seed=None):
        for epoch in tqdm(range(1, epochs + 1), ncols=70):
            # Reset Game
            self.reset()
            if seed is not None:
                self.env.seed(seed)
            state = self.env.reset()
            self.expand(None, state, 'root')

            while True:
                # Get Legal Actions
                legal_actions = self.env.get_legal_actions()
                if legal_actions is None:
                    # no place to a piece for all players
                    # TODO: Backpropagation
                    self.env.render()
                    break

                # Select & Expand
                action = self.get_action(self.cur_node, legal_actions)
                if action is None:  # it reached the end of the tree
                    break

                # Action!
                state, reward, done, info = self.env.step(action)
                # self.update_state(state, action)

                # Expand
                self.expand(self.cur_node, state, action)

                # Simulation
                self.simulate(self.env, state)

                if done:
                    break

        import ipdb
        ipdb.set_trace()

    def reset(self):
        self.history.clear()

        pass
