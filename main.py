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

    mcts = MCTS(env, simulation=1, max_depth=65, exploration_rate=0.1)
    mcts.train()


if __name__ == '__main__':
    main()
