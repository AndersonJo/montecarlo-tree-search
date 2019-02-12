import logging
import os
import pickle
from datetime import datetime
from math import sqrt
from random import random, choice
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
    def __init__(self, env: BaseEnv, c: float = sqrt(2),
                 max_depth: int = 50, max_depth_penalty: float = -1):
        self.env = env
        self.max_depth = max_depth
        self.max_depth_penalty = max_depth_penalty  # Tree reached the maximum depth, it will get penalty
        self.C = c  # Exploration constant

        self.nodes = dict()

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

    def get_best_action(self, env, state, legal_actions, debug=False):
        node: Node = self.nodes.get(state)
        if node is not None:
            action, _mode = self.get_action(node, legal_actions, 0, nodes=self.nodes)
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

            values = [(k, values[k]) for k in sorted(values, key=lambda k: -values[k])]

            if debug:
                for _k, _v in values:
                    print(_k, _v)

            action, value = values[0]
        return action, value

    def backpropagate(self, history, player, value, n_visit):
        total_value = 0
        count = 0
        for history_player, state in history[::-1]:
            node: Node = self.nodes[state]
            node.n_visit += n_visit
            if history_player != player:
                node.value += value
                total_value += node.value / node.n_visit
                count += 1
            else:
                node.value -= value

        return total_value, count

    def train(self, epochs: int = 10000, tie_value=None, exploration_rate: float = 0.9,
              checkpoint: str = 'checkpoint.pkl', seed=None):
        env = self.env.copy()
        _display_time = _save_time = datetime.now()

        _new_state_count = 0
        _backpropagation_count = 0
        _total_backpropagation = {}
        _turn_count = 0

        logger.info(f'tie_value:{tie_value} | C:{self.C} | exploration_rate:{exploration_rate} | seed:{seed} | '
                    f'checkpoint:{checkpoint}')

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
                                            exploration_rate=exploration_rate, nodes=self.nodes)
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
            if display_sec > 2:
                n_nodes = len(self.nodes)
                n_history = len(history)

                for k in _total_backpropagation:
                    _total_backpropagation[k] = round(_total_backpropagation[k], 1)
                print(f'nodes:{n_nodes} | new s:{_new_state_count} | hist:{n_history} | '
                      f'n:{_turn_count} | backpropagation:{_backpropagation_count}c')

                _display_time = datetime.now()
                _new_state_count = 0
                _backpropagation_count = 0
                _total_backpropagation = {}
                _turn_count = 0
                _total_value = 0

            if save_sec > 20:
                save(self, checkpoint)
                _save_time = datetime.now()
        else:
            save(self, checkpoint)

    def play(self, mode: str, exploration_rate: float = 0., seed: int = None):
        assert mode in ['cc', 'ch', 'hh', 'hc']

        # Set is_human fucntion
        def turn() -> bool:
            if not hasattr(turn, 'i'):
                turn.i = 0
            _is_human = mode[turn.i] == 'h'
            turn.i = abs(1 - turn.i)
            return _is_human

        is_human_turn = turn()

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
                action, value = self.get_best_action(env, state, legal_actions, debug=True)

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
    with open(file, 'wb') as f:
        pickle.dump(mcts, f)


def load(file='checkpoint.pkl') -> Union[MCTS, None]:
    if not os.path.exists(file):
        return None
    with open(file, 'rb') as f:
        mcts = pickle.load(f)

    logger.info('MCTS loaded succefully')
    return mcts
