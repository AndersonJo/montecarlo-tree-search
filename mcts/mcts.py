from copy import deepcopy
from math import sqrt
from random import random, choice
from typing import Union, List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from pprint import pprint
from mcts.env import BaseEnv


class Node:
    def __init__(self, action, value: float = 0., visit: int = 1):
        self.action = action
        self.value = value
        self.n_visit = visit
        self.children = dict()

    def has_child(self) -> bool:
        return bool(self.children)

    def add_child(self, state):
        self.children.setdefault(state, 0)
        self.children[state] += 1

    def __repr__(self):
        return '<Node.{0} w:{1} v:{2} c:{3}>'.format(self.action, self.value, self.n_visit, len(self.children))


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
            return self.nodes[self.history[-1][1]]

        return None

    @property
    def cur_depth(self):
        return len(self.history)

    def get_action(self, node: Node, legal_actions, nodes: Dict[str, int] = None) -> Union[None, Node]:
        """
        :return: next node's action
        """
        if self.cur_depth >= self.max_depth:
            return None

        if not node.has_child():  # reached the leaf node
            action = self.rand_action(node, legal_actions, nodes=nodes)
        elif random() <= self.exploration_rate and len(legal_actions) >= 1:
            action = self.rand_action(node, legal_actions, nodes=nodes)
        else:
            next_node, uct = self.calculate_uct(node, nodes=nodes)
            action = next_node.action

        return action

    def rand_action(self, node, legal_actions, nodes: Dict[str, int] = None) -> Node:
        assert len(legal_actions) >= 1
        if nodes is None:
            nodes = self.nodes

        tried_actions = {nodes[s] for s in node.children}
        action = choice(list(set(legal_actions) - tried_actions))
        return action

    def calculate_uct(self, node, nodes: Dict[str, int] = None) -> Tuple[Node, float]:
        if nodes is None:
            nodes = self.nodes

        nodes = [nodes[key] for key in node.children]
        ucts = map(lambda c: (c, (c.value / c.n_visit) + self.C * np.sqrt(np.log(node.n_visit) / c.n_visit)),
                   nodes)
        return sorted(ucts, key=lambda c: -c[1])[0]

    def expand(self, cur_node: Union[Node, None], state, action, history: list = None, nodes: Dict[str, int] = None,
               player=None):

        if nodes is None:
            nodes = self.nodes

        new_node = nodes.get(state, None)

        if new_node is None:
            new_node = Node(action=action)
            nodes[state] = new_node

        if cur_node is not None:
            cur_node.add_child(state)

        if history is not None and player is not None:
            history.append((player, state))
        return

    def simulate(self, env, state):

        value = 0
        for i in range(self.simulation):
            env_copied = env.copy()
            nodes = deepcopy(self.nodes)

            value += self._simulate(env_copied, state, nodes)
        return value

    def _simulate(self, env, state, nodes):
        player = env.player
        node = nodes[state]

        while True:

            legal_actions = env.get_legal_actions()
            if legal_actions is None:
                return env.calculate_reward_in_tie(player)
            assert len(legal_actions) > 0

            action = self.get_action(node, legal_actions, nodes=nodes)
            assert action in legal_actions

            if action is None:  # reached maximum depth of the tree
                return env.calculate_maximum_depth_penalty()

            state, reward, done, info = env.step(action)

            if done:
                return env.calculate_reward(player, reward, done)

            self.expand(node, state, action, nodes=nodes)
            node = nodes[state]

    def backpropagation(self, history, player, value):
        for history_player, state in history[::-1]:
            if history_player != player:
                node: Node = self.nodes[state]
                node.n_visit += self.simulation
                node.value += value

    def train(self, epochs: int = 10000, seed=None):
        for epoch in tqdm(range(1, epochs + 1), ncols=70):
            # Reset Game
            self.reset()
            if seed is not None:
                self.env.seed(seed)
            state = self.env.reset()
            self.expand(None, state, 'root', history=self.history, player=self.env.player)

            while True:
                # Get Legal Actions
                legal_actions = self.env.get_legal_actions()
                if legal_actions is None:
                    return self.env.calculate_reward_in_tie()

                assert len(legal_actions) > 0

                # Select & Expand
                action = self.get_action(self.cur_node, legal_actions)
                if action is None:  # it reached the end of the tree
                    break
                assert action in legal_actions

                # Action!
                state, reward, done, info = self.env.step(action)

                # Expand
                self.expand(self.cur_node, state, action, history=self.history, player=self.env.player)

                # Simulation
                value = self.simulate(self.env, state)

                # Backpropagation
                self.backpropagation(self.history, self.env.next_player(), value)

                if done:
                    break

    def reset(self):
        self.history.clear()

        pass
