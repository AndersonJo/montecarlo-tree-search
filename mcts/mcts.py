import logging
import os
import pickle
from datetime import datetime
from math import sqrt
from multiprocessing.pool import ThreadPool
from random import random, choice
from threading import Lock
from typing import Union, Dict

import numpy as np
from tqdm import tqdm

from mcts.env import BaseEnv

logger = logging.getLogger('mcts')


class Node:
    def __init__(self, value: float = 0., visit: int = 1):
        self.value = value
        self.n_visit = visit
        self.children = dict()

    def has_child(self) -> bool:
        return bool(self.children)

    def add_child(self, state, action):
        if state in self.children:
            assert self.children[state] == action
        else:
            self.children[state] = action

    def __repr__(self):
        return '<Node w:{0} v:{1} c:{2}>'.format(self.value, self.n_visit, len(self.children))


class MCTS(object):
    def __init__(self, env: BaseEnv, simulation: int = 10, c: float = sqrt(2),
                 exploration_rate: float = 0.5,
                 max_depth: int = 50, max_depth_penalty: float = -1):
        self.env = env
        self.simulation = simulation
        self.max_depth = max_depth
        self.max_depth_penalty = max_depth_penalty  # Tree reached the maximum depth, it will get penalty
        self.exploration_rate = exploration_rate
        self.C = c  # Exploration constant

        self.nodes = dict()

        # thread
        self.pool = ThreadPool(processes=self.simulation)
        self.lock = Lock()

    def get_action(self, node: Node, legal_actions, nodes: Dict[str, int] = None) -> Union[None, Node]:
        """
        :return: next node's action
        """
        # TODO
        # if len(self.history) >= self.max_depth:
        #     return None

        if node is None or nodes is None:
            action = choice(legal_actions)
        elif not node.has_child():  # reached the leaf node
            action = self.rand_action(node, legal_actions)
        elif random() <= self.exploration_rate and len(legal_actions) >= 1 and len(node.children) < len(legal_actions):
            action = self.rand_action(node, legal_actions)
        else:
            ucts = self.calculate_uct(self.C, node, nodes)
            action, next_node = ucts[0]
        return action

    def rand_action(self, node, legal_actions) -> Node:
        assert len(legal_actions) >= 1
        tried_actions = set(node.children.values())
        action = choice(list(set(legal_actions) - tried_actions))
        return action

    @staticmethod
    def calculate_uct(C, node, nodes: Dict[str, int]):
        action_nodes = [(node.children[key], nodes[key]) for key in node.children]
        ucts = map(lambda x: (x[0], (x[1].value / x[1].n_visit) + C * np.sqrt(np.log(node.n_visit) / x[1].n_visit)),
                   action_nodes)
        return sorted(ucts, key=lambda c: -c[1])

    def expand(self, cur_node: Union[Node, None], state: str, action, history: list = None,
               nodes: dict = None, player=None) -> bool:

        if nodes is None:
            nodes = self.nodes

        new_node = nodes.get(state, None)
        is_new = True if new_node is None else False

        if new_node is None:
            new_node = Node()
            nodes[state] = new_node

        if cur_node is not None:
            cur_node.add_child(state, action)

        if history is not None and player is not None:
            history.append((player, state))
        return is_new

    def simulate_actions(self, env, state, actions, simulation_depth: int = 100) -> dict:
        values = {}
        for action in actions:
            env_copied = env.copy()
            new_state, reward, done, info = env_copied.step(action)

            values[action] = self.simulate(env_copied, new_state, simulation_depth)

        values = [(k, values[k]) for k in sorted(values, key=lambda k: -values[k])]
        return values

    def simulate(self, env, state, simulation_depth: int = 4):

        value = 0
        results = []
        for i in range(self.simulation):
            env_copied = env.copy()
            args = (env_copied, state, simulation_depth)
            async_result = self.pool.apply_async(self._simulate, args)
            results.append(async_result)
            # value += self._simulate(env_copied, state, nodes)

        for async_result in results:
            value += async_result.get()
        return value

    def _simulate(self, env, state, depth=4):
        player = env.player
        node = self.nodes.get(state)

        i = 0
        while True:

            legal_actions, turn_changed = env.get_legal_actions()

            if legal_actions is None:
                return env.calculate_reward(player)
            assert len(legal_actions) > 0

            action = self.get_action(node, legal_actions)
            assert action in legal_actions

            if action is None:  # reached maximum depth of the tree
                return env.calculate_maximum_depth_penalty()

            new_state, reward, done, info = env.step(action)

            if done:
                return env.calculate_reward(player)

            node = self.nodes.get((state, new_state))
            state = new_state
            i += 1

            if i >= depth:
                if node is not None:
                    return node.value / node.n_visit
                return 0

    def backpropagation(self, history, player, value, n_visit):
        total_value = 0
        count = 0
        for history_player, state in history[::-1]:
            if history_player != player:
                node: Node = self.nodes[state]
                node.n_visit += n_visit
                node.value += value
                total_value += node.value / node.n_visit
                count += 1
        return total_value, count

    def train(self, epochs: int = 10000, simulation_depth: int = 4, seed=None):
        env = self.env.copy()

        logger.info(f'epoch:{epochs} | simulation_depth:{simulation_depth} | simulation:{self.simulation} |'
                    f' C:{self.C} | seed:{seed}')

        for epoch in tqdm(range(1, epochs + 1), ncols=50):
            # Log variables
            new_state_count = 0
            _time_simulation = 0
            _time_backpropagation = 0
            _backpropagation_count = 0
            _total_backpropagation = {}
            _turn_count = 0
            _total_value = 0

            # Reset Game
            history = []
            next_player = None

            if seed is not None:
                env.seed(seed)
            state = env.reset()
            current_player = env.player
            self.expand(None, state, 'root', history=history, player=current_player)

            while True:
                # Get Legal Actions
                legal_actions, turn_changed = env.get_legal_actions()
                if turn_changed:
                    current_player = env.player

                if legal_actions is None:
                    value = env.calculate_reward(current_player)
                    if value > 0:
                        back_value, back_count = self.backpropagation(history, current_player, value, 1)
                        _backpropagation_count += back_count
                        _total_backpropagation.setdefault(env.player, 0)
                        _total_backpropagation[env.player] += back_value
                    break
                assert len(legal_actions) > 0

                # Select & Expand
                action = self.get_action(self.nodes[history[-1][1]], legal_actions, nodes=self.nodes)
                if action is None:  # it reached the end of the tree
                    break
                assert action in legal_actions

                # Action!
                # After step function, player is changed.
                state, reward, done, info = env.step(action)
                next_player = env.player

                # Expand
                is_new = self.expand(self.nodes[history[-1][1]], state, action, history=history,
                                     player=next_player)

                if is_new:
                    new_state_count += 1

                # Simulation
                if _turn_count > 50:
                    _time_simulation_start = datetime.now()
                    value = self.simulate(env, state, simulation_depth=simulation_depth)
                    _total_value += value
                    _time_simulation += (datetime.now() - _time_simulation_start).total_seconds()

                    # Backpropagation
                    if value > 0:
                        back_value, back_count = self.backpropagation(history, current_player, value, self.simulation)
                        _backpropagation_count += back_count
                        _total_backpropagation.setdefault(current_player, 0)
                        _total_backpropagation[current_player] += back_value

                # Increase turn count
                _turn_count += 1

                # Gave Over
                if done:
                    back_value, back_count = self.backpropagation(history, current_player, reward, 1)
                    _backpropagation_count += back_count
                    _total_backpropagation.setdefault(current_player, 0)
                    _total_backpropagation[current_player] += back_value
                    break

                # Status Change
                current_player = next_player

            # Display
            if epoch % 2 == 0:
                n_nodes = len(self.nodes)
                n_history = len(history)
                _time_simulation = round(_time_simulation, 1)
                _total_value = round(_total_value, 1)
                for k in _total_backpropagation:
                    _total_backpropagation[k] = round(_total_backpropagation[k], 1)
                print(f'nodes:{n_nodes} | new s:{new_state_count} | hist:{n_history} | '
                      f'n:{_turn_count} | sim:{_total_value}v/{_backpropagation_count}c/{_time_simulation}s | '
                      f"w/b:{info['white']}/{info['black']}")

            if epoch % 2000 == 0:
                save(self)

    def play(self, mode: str, exploration_rate: float = 0.1, simulation=16, simulation_depth: int = 100,
             seed: int = None):
        assert mode in ['cc', 'ch', 'hh', 'hc']

        # Set is_human fucntion
        def turn() -> bool:
            if not hasattr(turn, 'i'):
                turn.i = 0
            _is_human = mode[turn.i] == 'h'
            turn.i = abs(1 - turn.i)
            return _is_human

        is_human_turn = turn()

        # Initialize parameters
        self.exploration_rate = exploration_rate
        self.simulation = simulation
        self.pool = ThreadPool(processes=self.simulation)

        # Initialize the Game
        env = self.env.copy()
        if seed is not None:
            env.seed(seed)
        state = env.reset()
        current_player = env.player
        env.render()

        while True:

            # Get Available Actions
            legal_actions, turn_changed = env.get_legal_actions()
            if turn_changed:
                current_player = env.player

            if legal_actions is None:
                logger.info('Tie! Game Over')
                break

            # Get Action
            if is_human_turn:
                env.render()
                action = env.play_as_human()
            else:
                action_values = self.simulate_actions(env, state, legal_actions, simulation_depth=simulation_depth)
                action, value = action_values[0]

                board = env.board.copy()
                board[:] = 0
                for _action, _value in action_values:
                    board[_action] = _value
                print(board)
                print('Decision:', action, 'value:', value, 'action_values:', action_values)

                # node: Node = self.nodes.get(state)
                # assert node is not None
                # logger.debug(f'children:{len(node.children)} | legal_actions: {len(legal_actions)}')
                # action = self.get_action(node, legal_actions, nodes=self.nodes)

            # Action!
            assert action is not None
            print('ACTION:', action)
            state, reward, done, info = env.step(action)

            # Render
            env.render()

            # Change Turn
            is_human_turn = turn()

            if done:
                print()
                print(f'Done | player:{env.player} | {info} | {reward}')
                break


def save(mcts: MCTS, file='checkpoint.pkl'):
    logger.info(f'saved {file}')
    mcts.pool = None
    with open(file, 'wb') as f:
        pickle.dump(mcts, f)
    mcts.pool = ThreadPool(processes=mcts.simulation)


def load(file='checkpoint.pkl') -> Union[MCTS, None]:
    if not os.path.exists(file):
        return None
    with open(file, 'rb') as f:
        mcts = pickle.load(f)

    mcts.pool = ThreadPool(processes=16)
    mcts.simulation = 4
    logger.info('MCTS loaded succefully')
    return mcts
