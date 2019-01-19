import argparse
import logging
import pickle
import random
from abc import ABC
from copy import deepcopy
from typing import List, Tuple

import gym
import numpy as np
from tqdm import tqdm

from games.othello import Othello

logger = logging.getLogger('mcts')


def parse_args():
    parser = argparse.ArgumentParser(description='MonteCarlo Tree Search Demo')
    parser.add_argument('--checkpoint', default='checkpoint.pkl', type=str)
    args = parser.parse_args()
    return args


class Node(object):
    def __init__(self, action, state=None, parent: 'Node' = None, depth=1):
        self.action = action
        self.state = state
        self.parent = parent
        self.n_win = 0
        self.n_visit = 1
        self.depth = depth
        self.children = list()
        self.info = dict()

    def has_child(self) -> bool:
        if self.children:
            return True
        return False

    def add_child(self, child_node) -> None:
        self.children.append(child_node)

    def calculate_uct(self, scalar) -> List[Tuple['Node', float]]:
        ucts = map(lambda c: (c, (c.n_win / c.n_visit) + scalar * np.sqrt(2 * np.log(self.n_visit) / c.n_visit)),
                   self.children)
        return sorted(ucts, key=lambda c: -c[1])

    def update(self, reward=0, visit=0):
        self.n_visit += visit
        self.n_win += reward

    @property
    def n_children(self):
        return len(self.children)

    def __str__(self):
        return '(Node.{0}-{1} w:{2} v:{3})'.format(self.state, self.action, self.n_win, self.n_visit)

    def __repr__(self):
        state = self.state
        if type(self.state) == np.ndarray:
            state = self.state.reshape(-1)
        return '(Node.{0}-{1} w:{2} v:{3})'.format(self.state, self.action, self.n_win, self.n_visit)


class MCTS(ABC):

    def __init__(self, env: Othello, actions=None, max_depth=20):
        self.env = env
        self._actions = actions
        self._temp_actions = None
        self._n_action = -1
        self.root = Node(action=None)
        self.cur_node = self.root
        self.max_depth = max_depth
        self.on_init()

    def get_actions(self, env=None, default_actions=None):
        if env is None:
            env = self.env

        if default_actions is not None:
            actions = self._temp_actions
        elif hasattr(env, 'get_legal_actions'):
            actions = env.get_legal_actions()
        else:
            actions = self._actions

        self._temp_actions = actions
        self._n_action = len(actions)
        assert actions is not None
        return actions

    def select(self, exploration_rate=0.5):
        if self.cur_node.depth >= self.max_depth:
            self.cur_node.n_win += -1
            return None

        cur_node = self.cur_node
        while True:
            if not cur_node.has_child():  # reached the leaf node
                break
            elif np.random.rand() <= exploration_rate and len(self.get_actions()) != cur_node.n_children:
                break

        return cur_node

    # def search_action2(self, state, exploration_rate=0.5):
    #     """
    #     Upper Confidence Bound
    #     """
    #     if self.cur_node.depth >= self.max_depth:
    #         self.cur_node.n_win += -1
    #         return None
    #
    #     if not self.cur_node.has_child():
    #         next_node = self.expand(self.cur_node, state)
    #     elif np.random.rand() <= exploration_rate and self.get_actions() != self.cur_node.n_children:
    #         next_node = self.expand(self.cur_node, state)
    #     else:
    #         next_node, uct = self.cur_node.calculate_uct(scalar=0.70)[0]
    #
    #     self.cur_node = next_node
    #     self._n_action = -1
    #     return self.cur_node.action
    #
    # def best_action(self):
    #     next_node, uct = self.cur_node.calculate_uct(scalar=0.1)[0]
    #     return next_node.action

    def expand(self, node, state, env=None, next=True) -> Node:
        if env is None:
            env = self.env

        actions = self.get_actions(env)
        assert len(actions) > len(node.children)

        tried_actions = {c.action for c in node.children}
        rand_action = random.choice(list(set(actions) - tried_actions))
        depth = self.cur_node.depth + 1
        new_node = Node(action=rand_action, state=state, parent=node, depth=depth)
        node.add_child(new_node)

        if next:
            self.cur_node = new_node

        self.on_expand(env, new_node, state)
        return new_node

    def simulate(self, start_node, state, n_simulation: int = 1, exploration_rate=0.8) -> int:
        total_reward = 0
        copied_node = deepcopy(start_node)

        for i in range(n_simulation):
            env = deepcopy(self.env)
            cur_node = copied_node

            while True:
                actions = self.get_actions(env)
                if len(actions) == 0:
                    # no place to put a piece for the current player
                    env.change_turn()
                    continue
                elif not cur_node.has_child():
                    next_node = self.expand(cur_node, state, env=env, next=False)
                elif np.random.rand() <= exploration_rate and len(actions) != cur_node.n_children:
                    next_node = self.expand(cur_node, state, env=env, next=False)
                else:
                    next_node, uct = cur_node.calculate_uct(scalar=0.70)[0]

                state, reward, done, info = env.step(next_node.action)
                self.after_simulation_step(self.env, env, start_node, cur_node, state, reward, done, info)

                if done:
                    final_reward = self.finish_simulation_step(i, self.env, env, start_node, cur_node, state,
                                                               reward, done, info, total_reward)
                    total_reward += final_reward
                    # logger.debug(f'{i} simulation done | reward:{final_reward} | total: {total_reward} '
                    #              f'| done:{done} | info:{info}')
                    break

                cur_node = next_node

        return total_reward

    def backpropagation(self, reward=0, visit=0):
        node = self.cur_node

        while node.parent is not None:
            reward = self.on_reward(self.env, node, reward, visit)
            node.update(reward, visit)
            node = node.parent

    def vanishing_backpropagation(self, reward):
        node = self.cur_node
        negative = False
        if reward < 0:
            negative = True
        while node.parent is not None:
            node.update(reward)
            reward = np.log1p(np.log1p(abs(reward)))
            if negative:
                reward *= -1
            node = node.parent

    def reset(self):
        self.cur_node = self.root
        self._n_action = -1
        self._temp_actions = None

    def display(self, node=None, step=0):
        if node is None:
            node = self.root

        print('{0} {1}'.format(' ' * step, node))
        for c in node.children:
            self.display(c, step + 1)

    def get_environment_name(self):
        raise NotImplementedError('get_environment_name not implemented error')

    def on_init(self):
        raise NotImplementedError('on_init not implemented')

    def on_expand(self, env, new_node: Node, state):
        raise NotImplementedError('on_expand not implemented')

    def after_simulation_step(self, start_env, simulation_env, start_node: Node, cur_node: Node,
                              state, reward, done, info):
        raise NotImplementedError('after_simulation_step not implemented')

    def finish_simulation_step(self, simulation_step: int, start_env, simulation_env, start_node: Node,
                               cur_node: Node, state, reward, done, info, total_reward: int) -> int:
        raise NotImplementedError('finish_simulation_step not implemented. it must return final score')

    def on_reward(self, env, node, reward, visit):
        raise NotImplementedError('on_reward not implemented.')


def save(mcts, seed=None, filename='checkpoint.pkl'):
    assert seed is not None
    with open(filename, 'wb') as f:
        pickle.dump({'mcts': mcts, 'seed': seed}, f)


def load(filename='checkpoint.pkl'):
    print('Loading {0}'.format(filename))
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    print(loaded_data)
    mcts, seed = loaded_data['mcts'], loaded_data['seed']
    print('Loading Done')
    return mcts, seed


def train(epochs, simulation, seed, checkpoint=None):
    if checkpoint is not None:
        mcts = load(checkpoint)
    else:
        mcts = MCTS(actions=list(ACTIONS.values()))
    taxi = gym.make('Taxi-v2')

    is_first_pickup = True
    pickup_count = 0
    success_count = 0
    wrong_pickup_count = 0
    end_count = 0
    for epoch in tqdm(range(1, epochs + 1), ncols=70):
        taxi.seed(seed)
        state = taxi.reset()
        mcts.reset()

        _do_simulation = True
        if epoch > simulation:
            _do_simulation = False

        while True:

            if _do_simulation:
                action = mcts.search_action(state, exploration_rate=1)
            else:
                action = mcts.search_action(state, exploration_rate=0.2)

            if action is None:  # It reached the end of the tree.
                mcts.backpropagation(visit=1)
                end_count += 1
                break

            next_state, reward, done, info = taxi.step(action)

            if action == ACTIONS['pickup'] and reward == -1:
                pickup_count += 1

                if is_first_pickup:
                    mcts.backpropagation(reward=10)
                    print('PICKED UP!')
                    print(taxi.render())
                    is_first_pickup = False

            if reward == -10:  # the taxi picked up the wrong person
                mcts.backpropagation(visit=1)
                wrong_pickup_count += 1
                break

            if reward > 0:  # The game has been solved!
                success_count += 1
                mcts.backpropagation(reward=2, visit=1)
                break

            if done:
                break

        if epoch % 50000 == 0:
            # mcts.display()
            save(mcts, seed, filename='checkpoint.pkl')

            print(' - pickup:{0} success:{1}, wrong_pickup:{2}, end_tree:{3}'.format(
                pickup_count, success_count, wrong_pickup_count, end_count))

            if wrong_pickup_count == 0 and success_count == 50000:
                print('Early Stop!')
                print('Training Successfully Done')
                break

            pickup_count = 0
            success_count = 0
            wrong_pickup_count = 0
            end_count = 0

            # demo(taxi, mcts, seed=seed)

    # demo(taxi, mcts, seed=seed)


def demo(taxi, mcts, seed=2):
    taxi.seed(seed)
    state = taxi.reset()
    mcts.reset()

    while True:
        action = mcts.search_action(state, exploration_rate=0.0)
        if action is None:
            break
        state, reward, done, info = taxi.step(action)
        print(taxi.render(), 'action:', action, 'reward:', reward)

        if done:
            break


def test():
    args = parse_args()
    mcts, seed = load(args.checkpoint)
    taxi = gym.make('Taxi-v2')
    demo(taxi, mcts, seed=seed)


if __name__ == '__main__':
    seed = np.random.randint(0, 50000)
    np.random.seed(seed)
    seed = 38260
    print('SEED:', seed)
    train(epochs=30000000, simulation=15000000, seed=seed)
