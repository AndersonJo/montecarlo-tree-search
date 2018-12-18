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
        return len(list(filter(lambda c: c[1] != -1, self.children)))

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
        self.cur_action = None

    def init_root_state(self, state):
        self.cur_state = state
        self.cur_action = -1
        if state not in self.nodes:
            self.nodes[state][-1] = Node(state=state, action=-1)
        elif -1 not in self.nodes[state]:
            self.nodes[state][-1] = Node(state=state, action=-1)
        self.root_state = state

    def get_current_node(self):
        return self.nodes[self.cur_state][self.cur_action]

    def search_action(self, state):
        """
        Additionally it updates the current node by updating cur_state and cur_action
        :param state: Current state
        :return:
        """
        cur_node = self.get_current_node()
        if cur_node.n_children == 0:
            next_node = self.expand(state)

        elif np.random.rand() >= 0.2 and self.n_actions > cur_node.n_children:  # more exploration is required
            next_node = self.expand(state)
        else:
            next_node, uct = self.calculate_ucb(scalar=0.7)

        # Update current state and action
        self.cur_state = next_node.state
        self.cur_action = next_node.action
        # assert self.cur_state in self.nodes
        # assert self.cur_action in self.nodes[self.cur_state]

        return next_node, next_node.action

    def best_action(self):
        next_node, uct = self.cur_node.calculate_uct(scalar=0.1)[0]
        return next_node.action

    def expand(self, state):
        cur_node = self.get_current_node()
        assert self.n_actions > cur_node.n_children

        tried_actions = set([c[1] for c in cur_node.children])
        rand_action = random.choice(list(set(self.actions) - tried_actions))

        if state in self.nodes and rand_action in self.nodes[state]:
            existing_node: Node = self.nodes[state][rand_action]
            children_states = [c[0] for c in existing_node.children]

            if self.cur_state in children_states:
                next_node = Node(state=state, action=rand_action, parent=(self.cur_state, self.cur_action))
                rand_action = (rand_action, hash(next_node))
                self.nodes[state][rand_action] = next_node
            else:
                next_node = existing_node
        else:
            next_node = Node(state=state, action=rand_action, parent=(self.cur_state, self.cur_action))
            self.nodes[state][rand_action] = next_node

        cur_node.add_child(state=state, action=rand_action)
        # assert state in self.nodes
        # assert rand_action in self.nodes[state]
        return next_node

    def calculate_ucb(self, scalar):
        """
        Upper Confidence Bound
        The algorithm selects a node that maximize some quality.

        @param cur_state: current state
        @param scalar: scalar is a weight for exploration over exploitation
        """
        cur_node = self.get_current_node()

        keys = list(self.nodes[self.cur_state].keys())
        if -1 in keys:
            keys.remove(-1)
        children = [self.nodes[self.cur_state][a] for a in keys]

        ucts = map(lambda c: (c, (c.n_win / c.n_visit) + scalar * np.sqrt(2 * np.log(cur_node.n_visit) / c.n_visit)),
                   children)

        top_node = sorted(ucts, key=lambda c: -c[1])[0]
        return top_node

    def backpropagation(self, reward):
        node = self.get_current_node()
        count = 0
        while node.parent is not None:
            node.update(reward)
            node = self.nodes[node.parent[0]][node.parent[1]]
            count += 1
        return count

    def display(self, state=None, action=None, step=0, _visits={}):
        if state is None:
            state = self.root_state
            action = -1

        node: Node = self.nodes[state][action]
        # _visits[(state, action)] = True
        if node.parent:
            parent = self.nodes[node.parent[0]][node.parent[1]]
            self.display(parent.state, parent.action, step + 1, _visits=_visits)
        print('{0} {1}'.format(' ' * step, node))


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

        backpropagation_count = 0
        count = 0
        longest = 0
        longest_node = None

        while True:
            next_node, action = mcts.search_action(state=state)
            state, reward, done, info = taxi.step(action)

            if count > longest:
                longest_node = next_node

            if reward == -10:
                backpropagation_count += mcts.backpropagation(reward=reward)
                fail_pickup_count += 1
                break

            if reward > 0:
                backpropagation_count += mcts.backpropagation(reward=reward)
                success_count += 1
                break

            if done:
                break

            count += 1

        if epoch % 10000 == 0:
            if longest_node:
                mcts.display(longest_node.state, longest_node.action)
            save(mcts)
            backpropagation_count /= 10000
            total = sum([len(node) for node in mcts.nodes.values()])
            print(' - O:{0}, X:{1}, backpropagation:{2}, nodes:{3} total:{4}'.format(
                success_count, fail_pickup_count, backpropagation_count, len(mcts.nodes), total))

    # demo(taxi, mcts)


def demo(taxi, mcts: MCTS):
    state = taxi.reset()
    mcts.init_root_state(state)

    while True:
        action = mcts.search_action()
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
