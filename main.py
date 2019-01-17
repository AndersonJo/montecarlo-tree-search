import argparse
import logging
from argparse import Namespace

import gym
import numpy as np
from tqdm import tqdm

from games import register_games
from games.othello import Othello
from mcts.env import BaseEnv
from mcts.mcts import MCTS

GAMES = {'taxi': 'Taxi-v2',
         'othello': 'othello-v0'}

logger = logging.getLogger('mcts')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s | %(levelname)s] %(message)s'))
logger.addHandler(ch)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='MonteCarlo Tree Search Demo')
    parser.add_argument('game', type=str)
    parser.add_argument('--simulation', default=1, type=str)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--checkpoint', default='checkpoint.pkl', type=str)
    args = parser.parse_args()

    assert args.game.lower() in GAMES
    args.game = GAMES[args.game.lower()]

    return args


# def train(env: Othello, mcts: MCTS, epochs: int, n_simulation: int, seed: int = None, checkpoint: str = None):
#     # is_first_pickup = True
#     # pickup_count = 0
#     # success_count = 0
#     # wrong_pickup_count = 0
#     # end_count = 0
#
#     for epoch in tqdm(range(1, epochs + 1), ncols=70):
#         # Reset Game
#         if seed is not None:
#             env.seed(seed)
#         state = env.reset()
#         mcts.reset()
#
#         while True:
#             # Select
#             leaf_node = mcts.select()
#             if leaf_node is None:  # It reached the end of the tree or overcame the limitation
#                 # TODO: backpropagation for reaching end of the game without success
#                 break
#
#             # Expand
#             next_node = mcts.expand(leaf_node, state)
#             state, reward, done, info = env.step(next_node.action)
#
#             # Simulation
#             print('BEFORE SIMULATION', env.PLAYERS[env.turn])
#             simulation_score = mcts.simulate(next_node, state, n_simulation=n_simulation)
#
#             # Backpropagation
#             mcts.backpropagation(reward=simulation_score, visit=n_simulation)
#
#             if done:
#                 break
#
#         if epoch % 50000 == 0:
#             demo(env, mcts, seed=seed)
#             break
#         break


def demo(env, mcts, seed=2):
    env.seed(seed)
    state = env.reset()
    mcts.reset()

    while True:
        action = mcts.search_action(state, exploration_rate=0.0)
        if action is None:
            break
        state, reward, done, info = taxi.step(action)
        print(taxi.render(), 'action:', action, 'reward:', reward)

        if done:
            break


def main():
    # Register Games
    register_games()

    # Parse Arguments
    args = parse_args()

    # Set Random Seed
    seed = np.random.randint(0, 50000)
    np.random.seed(seed)
    seed = 38260

    # Get Environment and Set Actions
    env: BaseEnv = gym.make(args.game)

    mcts = MCTS(env, max_depth=65)
    mcts.train()


if __name__ == '__main__':
    main()
