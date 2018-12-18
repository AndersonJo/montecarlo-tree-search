import pickle
from typing import List, Tuple, Union

import gym
import numpy as np
import random

from tqdm import tqdm

ACTIONS = {'left': 3, 'right': 2, 'up': 1, 'down': 0, 'pickup0': 4, 'dropoff': 5}


class Node(object):
    def __init__(self, action: Union[None, int], state=None, parent: 'Node' = None):
        self.action = action
        self.state = state
        self.parent = parent
        self.n_win = 0
        self.n_visit = 1
        self.children = list()

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

    def update(self, reward):
        self.n_visit += 1
        self.n_win += reward

    @property
    def n_children(self):
        return len(self.children)

    def __str__(self):
        return '(Node.{0}-{1} w:{2} v:{3})'.format(self.state, self.action, self.n_win, self.n_visit)

    def __repr__(self):
        return '(Node.{0}-{1} w:{2} v:{3})'.format(self.state, self.action, self.n_win, self.n_visit)


class MCTS(object):

    def __init__(self, actions, use_state=True):
        self.actions = actions
        self.n_actions = len(actions)
        self.root = Node(action=None)
        self.cur_node = self.root
        self.use_state = use_state

    def search_action(self, state):
        """
        Upper Confidence Bound
        """
        if not self.cur_node.has_child():
            next_node = self.expand(self.cur_node, state)

        elif np.random.rand() >= 0.2 and self.n_actions != self.cur_node.n_children:  # more exploration is required
            next_node = self.expand(self.cur_node, state)
        else:
            next_node, uct = self.cur_node.calculate_uct(scalar=0.70)[0]

        self.cur_node = next_node
        return self.cur_node.action

    def best_action(self):
        next_node, uct = self.cur_node.calculate_uct(scalar=0.1)[0]
        return next_node.action

    def expand(self, node, state) -> Node:
        assert self.n_actions > len(node.children)

        tried_actions = {c.action for c in node.children}
        rand_action = random.choice(list(set(self.actions) - tried_actions))
        new_node = Node(action=rand_action, state=state, parent=node)
        node.add_child(new_node)
        return new_node

    def backpropagation(self, reward):
        node = self.cur_node
        while node.parent is not None:
            node.update(reward)
            node = node.parent

    def reset(self):
        self.cur_node = self.root

    def display(self, node=None, step=0):
        if node is None:
            node = self.root

        print('{0} {1}'.format(' ' * step, node))
        for c in node.children:
            self.display(c, step + 1)


def save(mcts, filename='checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(mcts, f)


def load(filename='checkpoint.pkl') -> MCTS:
    print('Loading {0}'.format(filename))
    with open(filename, 'rb') as f:
        mcts = pickle.load(f)
        mcts.cur_node = mcts.root
    print('Loading Done')
    return mcts


def train(epochs=700000):
    mcts_set = {}
    taxi = gym.make('Taxi-v2')

    success_count = 0
    fail_pickup_count = 0

    for epoch in tqdm(range(epochs), ncols=70):
        state = taxi.reset()
        mcts_set.setdefault(state, MCTS(actions=list(ACTIONS.values())))
        mcts = mcts_set[state]
        mcts.reset()

        while True:
            action = mcts.search_action(state)

            state, reward, done, info = taxi.step(action)

            if reward == -10:
                mcts.backpropagation(-1)
                fail_pickup_count += 1
                break

            if reward > 0:
                success_count += 1
                mcts.backpropagation(reward)
                break

            if done:
                break

        if epoch % 50000 == 0:
            # mcts.display()
            save(mcts_set)

            print(' - success:{0}, wrong_pickup:{1}'.format(success_count, fail_pickup_count))

    demo(taxi, mcts_set)


def demo(taxi, mcts_set):
    state = taxi.reset()
    mcts = mcts_set[state]
    mcts.reset()

    while True:
        action = mcts.best_action()
        state, reward, done, info = taxi.step(action)
        print(taxi.render(), 'action:', action, 'reward:', reward)

        if done:
            break


def test():
    mcts = load()
    taxi = gym.make('Taxi-v2')

    state = taxi.reset()
    mcts.reset()
    reward = 0

    while True:
        action = mcts.best_action()
        state, reward, done, info = taxi.step(action)
        print(taxi.render(), 'action:', action, 'reward:', reward)
        input()
        if done:
            break

    if reward > 10:
        mcts.display()

        print('score: {0}'.format(reward))


if __name__ == '__main__':
    train()

    # test()
