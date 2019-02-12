import logging
import os
import pickle
from datetime import datetime
from math import sqrt
from multiprocessing.pool import ThreadPool
from pprint import pprint
from random import random, choice
from threading import Lock
from time import sleep
from typing import Union, Dict, Tuple

import numpy as np
from tqdm import tqdm

from mcts.env import BaseEnv

logger = logging.getLogger('mcts')


class Node:
    def __init__(self, value: float = 1., visit: int = 1):
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

    def get_action(self, node: Node, legal_actions, exploration_rate: float,
                   nodes: Dict[str, int] = None) -> Tuple[Node, str]:
        """
        :return: next node's action
        """
        # TODO
        # if len(self.history) >= self.max_depth:
        #     return None

        if node is None or nodes is None:
            action = choice(legal_actions)
            mode = 'random_choice_no_history'
        elif not node.has_child():  # reached the leaf node
            action = self.rand_action(node, legal_actions)
            mode = 'random_choice_no_child'
        elif random() <= exploration_rate and len(legal_actions) >= 1 and len(node.children) < len(legal_actions):
            action = self.rand_action(node, legal_actions)
            mode = 'exploration'
        else:
            ucts = self.calculate_uct(self.C, node, nodes)
            # for _a, _n in ucts:
            #     print(f'UCT action:{_a} | {_n}')
            action, next_node = ucts[0]
            mode = 'calculate_uct'
        return action, mode

    def rand_action(self, node, legal_actions) -> Node:
        assert len(legal_actions) >= 1
        tried_actions = set(node.children.values())
        action = choice(list(set(legal_actions) - tried_actions))
        return action

    @staticmethod
    def calculate_uct(c, node, nodes: Dict[str, int]):
        action_nodes = [(node.children[key], nodes[key]) for key in node.children]
        ucts = map(lambda x: (x[0], (x[1].value / x[1].n_visit) + c * np.sqrt(np.log(node.n_visit) / x[1].n_visit)),
                   action_nodes)
        return sorted(ucts, key=lambda c: -c[1])

    def expand(self, cur_node: Union[Node, None], state: str, action, history: list = None,
               nodes: dict = None, player=None) -> bool:
        is_new = False
        if nodes is None:
            nodes = self.nodes

        if state not in nodes:
            nodes[state] = Node()
            is_new = True

        if cur_node is not None:
            cur_node.add_child(state, action)

        if history is not None and player is not None:
            history.append((player, state))
        return is_new

    def get_best_action(self, env, state, legal_actions, simulation_depth: int = 100, asynchronous: bool = False,
                        debug=False):
        # values = {}
        # player = env.init_player
        #
        # for action in legal_actions:
        #     env_copied = env.copy()
        #     new_state, reward, done, info = env_copied.step(action)
        #     values[action] = self.simulate(env_copied, new_state, player, 1, simulation_depth, asynchronous)
        #
        # values = [(k, values[k]) for k in sorted(values, key=lambda k: -values[k])]
        # for k, v in values:
        #     print(k, v)
        #
        # action, value = values[0]
        # return action, value

        node: Node = self.nodes.get(state)
        if node is not None:
            action, _mode = self.get_action(node, legal_actions, self.exploration_rate, nodes=self.nodes)
            value = node.value / node.n_visit
            if debug:
                for _action in legal_actions:
                    env_copied = env.copy()
                    new_state, reward, done, info = env_copied.step(_action)
                    node: Node = self.nodes.get(new_state)
                    if node is None:
                        continue

                    print(f'get_best_action <node> | action:{_action} | value:{node.value} | visit: {node.n_visit} | '
                          f'score: {node.value / node.n_visit:.2}')
        else:
            values = {}
            player = env.init_player
            for action in legal_actions:
                env_copied = env.copy()
                new_state, reward, done, info = env_copied.step(action)
                node: Node = self.nodes.get(new_state)
                if node is None:
                    continue

                values[action] = node.value / node.n_visit

                # values[action] = self.simulate(env_copied, new_state, player, simulation_depth)

            values = [(k, values[k]) for k in sorted(values, key=lambda k: -values[k])]

            if debug:
                for _k, _v in values:
                    print(_k, _v)

            action, value = values[0]
        return action, value

    def simulate(self, env, state, player, simulation: int = 1, simulation_depth: int = 100,
                 asynchronous: bool = False):
        value = 0
        results = []

        if asynchronous:
            for i in range(simulation):
                env_copied = env.copy()
                args = (env_copied, state, player, simulation_depth)
                async_result = self.pool.apply_async(self._simulate, args)
                results.append(async_result)

            for async_result in results:
                value += async_result.get()
        else:
            for i in range(simulation):
                env_copied = env.copy()
                env_copied.player = player
                args = (env_copied, state, player, simulation_depth)
                value += self._simulate(*args)

        return value

    def _simulate(self, env: BaseEnv, state, player: int, depth: int = 4, tie_value: int = 0):
        assert player == env.init_player
        node = self.nodes.get(state)

        # env.render()
        # print(f'state:{state} | env.init_player:{env.init_player} , cur_player:{env.cur_player}, player:{player}')

        i = 0
        while True:
            legal_actions = env.get_legal_actions()

            if legal_actions is None or not legal_actions:
                return tie_value
                # return env.calculate_reward(player)
            assert len(legal_actions) > 0

            action, _ = self.get_action(node, legal_actions, exploration_rate=0, nodes=self.nodes)
            if action is None:  # reached maximum depth of the tree
                return 0
            assert action in legal_actions

            new_state, reward, done, info = env.step(action)

            # env.render()
            # print(env.board)
            # print(
            #     f'cur_player:{env.cur_player}, player:{player} | action:{action} | reward:{reward} | done:{done} | mode:{_} | node:{node}')

            if done:
                return reward

            node = self.nodes.get((state, new_state))
            state = new_state
            i += 1

            if i >= depth:
                if node is not None:
                    return node.value / node.n_visit
                return 0

    def backpropagate(self, history, player, value, n_visit):
        total_value = 0
        count = 0
        for history_player, state in history[::-1]:
            node: Node = self.nodes[state]
            node.n_visit += n_visit
            if history_player != player:
                # print(f'backpropagation | history_player:{history_player} | player:{player} | state:{state}')
                node.value += value
                total_value += node.value / node.n_visit
                count += 1
            else:
                node.value -= value

        return total_value, count

    def train(self, epochs: int = 10000, simulation_depth: int = 4, tie_value=None, checkpoint: str = 'checkpoint.pkl',
              asynchronous: bool = False, seed=None):
        env = self.env.copy()
        _display_time = _save_time = datetime.now()

        _new_state_count = 0
        _backpropagation_count = 0
        _total_backpropagation = {}
        _time_simulation = 0
        _turn_count = 0
        _total_value = 0

        logger.info(f'epoch:{epochs} | simulation_depth:{simulation_depth} | simulation:{self.simulation} |'
                    f' C:{self.C} | seed:{seed}')

        for epoch in tqdm(range(1, epochs + 1), ncols=70):
            # Log variables

            # Reset Game
            history = []

            if seed is not None:
                env.seed(seed)
            state = env.reset()
            init_player = env.init_player
            self.expand(None, state, 'root', history=history, player=env.cur_player)

            while True:
                # Get Legal Actions
                legal_actions = env.get_legal_actions()

                if legal_actions is None or not legal_actions:
                    if tie_value is not None and tie_value != 0:
                        back_value, back_count = self.backpropagate(history, init_player, tie_value, 1)
                        _backpropagation_count += back_count
                        _total_backpropagation.setdefault(env.init_player, 0)
                        _total_backpropagation[env.init_player] += back_value
                    break
                assert len(legal_actions) > 0

                # Select & Expand
                action, _ = self.get_action(self.nodes[history[-1][1]], legal_actions,
                                            exploration_rate=self.exploration_rate, nodes=self.nodes)
                if action is None:  # it reached the end of the tree
                    break
                assert action in legal_actions

                # Action!
                # After step function, player is changed.
                state, reward, done, info = env.step(action)
                assert init_player == env.init_player

                # Expand
                is_new = self.expand(self.nodes[history[-1][1]], state, action, history=history,
                                     player=env.cur_player)

                if is_new:
                    _new_state_count += 1

                # Simulation
                # if _turn_count > 50:
                #     _time_simulation_start = datetime.now()
                #     value = self.simulate(env, state, player=init_player, simulation_depth=simulation_depth,
                #                           asynchronous=asynchronous)
                #     _total_value += value
                #     _time_simulation += (datetime.now() - _time_simulation_start).total_seconds()
                #
                #     # Backpropagation
                #     if value != 0:
                #         back_value, back_count = self.backpropagate(history, current_player, value, self.simulation)
                #         _backpropagation_count += back_count
                #         _total_backpropagation.setdefault(current_player, 0)
                #         _total_backpropagation[current_player] += back_value

                # Increase turn count
                _turn_count += 1

                # Gave Over

                if done:
                    back_value, back_count = self.backpropagate(history, init_player, reward, 1)

                    _backpropagation_count += back_count
                    _total_backpropagation.setdefault(init_player, 0)
                    _total_backpropagation[init_player] += back_value
                    break

            # Display
            display_sec = (datetime.now() - _display_time).total_seconds()
            save_sec = (datetime.now() - _save_time).total_seconds()
            if display_sec >= 1:
                n_nodes = len(self.nodes)
                n_history = len(history)
                _time_simulation = round(_time_simulation, 1)
                _total_value = round(_total_value, 1)
                for k in _total_backpropagation:
                    _total_backpropagation[k] = round(_total_backpropagation[k], 1)
                print(f'nodes:{n_nodes} | new s:{_new_state_count} | hist:{n_history} | '
                      f'n:{_turn_count} | sim:{_total_value}v/{_backpropagation_count}c/{_time_simulation}s')

                _display_time = datetime.now()
                _new_state_count = 0
                _backpropagation_count = 0
                _total_backpropagation = {}
                _time_simulation = 0
                _turn_count = 0
                _total_value = 0

            if save_sec >= 20:
                save(self, checkpoint)
                _save_time = datetime.now()
        else:
            save(self, checkpoint)

    def play(self, mode: str, exploration_rate: float = 0., simulation=512, simulation_depth: int = 100,
             asynchronous: bool = False, seed: int = None):
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
        self.pool = ThreadPool(processes=simulation)

        # Initialize the Game
        env = self.env.copy()
        if seed is not None:
            env.seed(seed)
        state = env.reset()
        env.render()

        while True:
            # Get Available Actions
            legal_actions = env.get_legal_actions()

            if legal_actions is None:
                logger.info('Tie! Game Over')
                break
            print()
            print('[TURN]=======================================================')
            print(f'Current Player:{env.PLAYERS[env.init_player]} {env.init_player}')
            print(env.board)

            # Get Action
            if is_human_turn:
                env.render()
                action = env.play_as_human()
                if action is None:
                    break
            else:
                action, value = self.get_best_action(env, state, legal_actions, simulation_depth=simulation_depth,
                                                     debug=True)

                print(f'Computer:{env.init_player} | Decision: {action} | value:{value}')

                # node: Node = self.nodes.get(state)
                # assert node is not None
                # logger.debug(f'children:{len(node.children)} | legal_actions: {len(legal_actions)}')
                # action, _mode = self.get_action(node, legal_actions, nodes=self.nodes)
                # print('[Get Action]')
                # print(f'action:{action} | mode:{_mode}')

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
                print(f'[done] | player:{env.init_player} | reward:{reward}')
                break

            if mode == 'cc':
                sleep(0.5)
        sleep(1)


def save(mcts: MCTS, file='checkpoint.pkl'):
    logger.info(f'saved {file}')
    mcts.pool = None
    mcts.lock = None
    with open(file, 'wb') as f:
        pickle.dump(mcts, f)
    mcts.pool = ThreadPool(processes=mcts.simulation)
    mcts.lock = Lock()


def load(file='checkpoint.pkl') -> Union[MCTS, None]:
    if not os.path.exists(file):
        return None
    with open(file, 'rb') as f:
        mcts = pickle.load(f)

    mcts.pool = ThreadPool(processes=16)
    mcts.simulation = 4
    logger.info('MCTS loaded succefully')
    return mcts
