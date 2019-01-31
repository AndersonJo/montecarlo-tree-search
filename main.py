import argparse
import logging
from argparse import Namespace

import gym
import numpy as np
from tqdm import tqdm

from games import register_games
from games.othello import Othello
from mcts.env import BaseEnv
from mcts.mcts import MCTS, load

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
    parser.add_argument('mode', type=str, choices=['train', 'play'])
    parser.add_argument('game', type=str, choices=['othello', 'taxi'])
    parser.add_argument('--versus', type=str, choices=[''])
    parser.add_argument('--simulation', default=3, type=str)
    parser.add_argument('--exploration', default=0.9, type=float)
    parser.add_argument('--epochs', default=900000, type=int)
    parser.add_argument('--checkpoint', default='checkpoint.pkl', type=str)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    args.mode = args.mode.lower()

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
    mcts = load()

    if args.mode == 'train':
        if mcts is None:
            mcts = MCTS(env, simulation=args.simulation, max_depth=100, exploration_rate=args.exploration)
        mcts.exploration_rate = args.exploration
        mcts.simulation_depth = 100
        mcts.train(epochs=args.epochs, simulation_depth=70, seed=args.seed)
    if args.mode == 'play':
        mcts.play(mode='hc')
    print('DONE!!')


if __name__ == '__main__':
    main()
