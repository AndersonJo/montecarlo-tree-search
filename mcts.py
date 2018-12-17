import pickle
from typing import List, Tuple, Union, Dict

import gym
import numpy as np
import random

from tqdm import tqdm
from collections import defaultdict

ACTIONS = {'left': 3, 'right': 2, 'up': 1, 'down': 0, 'pickup0': 4, 'dropoff': 5}


class Node(object):
    def __init__(self, state=None, action: Union[None, int] = -1, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.n_win = 0
        self.n_visit = 1
        self.children = list()

    def has_child(self) -> bool:
        if self.children:
            return True
        return False

    def add_child(self, state, action) -> None:
        if (state, action) not in self.children:
            self.children.append((state, action))

    def update(self, reward):
        self.n_visit += 1
        self.n_win += reward

    @property
    def n_children(self):
        return len(self.children)

    def __str__(self):
        return '(Node {0}->{1} actions:{2} win:{3} visit:{4})'.format(
            self.parent, self.state, self.children, self.n_win, self.n_visit)

    def __repr__(self):
        return '(Node {0}->{1} actions:{2} win:{3} visit:{4})'.format(
            self.parent, self.state, self.children, self.n_win, self.n_visit)


class MCTS(object):

    def __init__(self, actions):
        self.actions = actions
        self.n_actions = len(actions)

        # Create Nodes
        self.nodes: Dict[Dict[Node]] = defaultdict(dict)
        self.root_state = None
        self.cur_state = None

    def init_root_state(self, state):
        self.cur_state = state
        if state not in self.nodes:
            self.nodes[state][-1] = Node(state=state, action=-1)
        self.root_state = state

    def search_action(self, state, action):
        cur_node = self.nodes[self.cur_state][action]

        if not cur_node.children:
            next_node = self.expand(self.cur_state, state, action)

        elif np.random.rand() >= 0.2 and self.n_actions != cur_node.n_children:  # more exploration is required
            next_node = self.expand(self.cur_state, state, action)
        else:
            print('calculate_ucb', self.nodes[self.cur_state])
            print('mnnnnnnnsssss', self.nodes[state])
            next_node, uct = self.calculate_ucb(self.cur_state, action=action, scalar=0.7)

        self.cur_state = state
        return next_node, next_node.action

    def best_action(self):
        next_node, uct = self.cur_node.calculate_uct(scalar=0.1)[0]
        return next_node.action

    def expand(self, cur_state, state, action):
        cur_node = self.nodes[cur_state][action]
        assert self.n_actions > cur_node.n_children

        tried_actions = set([c[1] for c in cur_node.children])
        rand_action = random.choice(list(set(self.actions) - tried_actions))

        if state in self.nodes and rand_action in self.nodes[state]:
            next_node = self.nodes[state][rand_action]
        else:
            next_node = Node(state=state, action=rand_action, parent=(cur_state, action))
            self.nodes[state][rand_action] = next_node

        cur_node.add_child(state=state, action=rand_action)
        return next_node

    def calculate_ucb(self, cur_state, action, scalar):
        """
        Upper Confidence Bound
        The algorithm selects a node that maximize some quality.

        @param cur_state: current state
        @param scalar: scalar is a weight for exploration over exploitation
        """
        actions = self.nodes[cur_state]
        cur_node = self.nodes[cur_state][action]

        keys = filter(lambda k: k != -1, actions.keys())
        children = [self.nodes[cur_state][a] for a in keys]

        ucts = map(lambda c: (c, (c.n_win / c.n_visit) + scalar * np.sqrt(2 * np.log(cur_node.n_visit) / c.n_visit)),
                   children)
        ucts = filter(lambda c: c[0].action != -1, ucts)

        try:
            sorted_ucts = sorted(ucts, key=lambda c: -c[1])[0]
        except:
            import ipdb
            ipdb.set_trace()
        return sorted_ucts

    def backpropagation(self, state, action, reward):
        node = self.nodes[state][action]

        while node.parent is not None:
            node.update(reward)
            node = self.nodes[node.parent[0]][node.parent[1]]

    def display(self, state=None, action=None, step=0):
        if state is None:
            state = self.root_state
            action = -1

        node: Node = self.nodes[state][action]
        print('{0} {1}'.format(' ' * step, node))
        for action in node.children:
            self.display(node.state, action, step + 1)


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
    taxi = gym.make('Taxi-v2')
    init_state = taxi.reset()

    mcts = MCTS(actions=list(ACTIONS.values()))
    mcts.init_root_state(init_state)

    success_count = 0
    fail_pickup_count = 0

    for epoch in tqdm(range(epochs), ncols=70):
        init_state = state = taxi.reset()
        mcts.init_root_state(init_state)
        action = -1
        print('START', mcts.nodes)

        while True:

            next_node, action = mcts.search_action(state=state, action=action)
            print('wwwwwwwwww', next_node, action)
            # if action == -1:
            #     import ipdb
            #     ipdb.set_trace()
            state, reward, done, info = taxi.step(next_node.action)

            if reward == -10:
                mcts.backpropagation(state=state, action=action, reward=-1)
                fail_pickup_count += 1
                break

            if reward > 0:
                success_count += 1
                mcts.backpropagation(state=state, action=action, reward=reward)
                break

            if done:
                break

        if epoch % 50000 == 0:
            mcts.display(state, action)
            save(mcts)

            print(' - O:{0}, X:{1} nodes:{2}'.format(success_count, fail_pickup_count, len(mcts.nodes)))

    # demo(taxi, mcts)


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
