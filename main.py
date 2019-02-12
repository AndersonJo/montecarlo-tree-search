import argparse
import logging
from argparse import Namespace

import gym
import numpy as np

from games import register_games
from mcts.env import BaseEnv
from mcts.mcts import MCTS, load

GAMES = {'taxi': {'name': 'Taxi-v2',
                  'epochs': 50000},
         'othello': {'name': 'Othello-v0',
                     'epochs': 90000},
         'tictactoe': {'name': 'TicTacToe-v0',
                       'epochs': 50000,
                       'checkpoint': 'tictactoe.pkl',
                       'tie': 0}}

logger = logging.getLogger('mcts')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s | %(levelname)s] %(message)s'))
logger.addHandler(ch)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='MonteCarlo Tree Search Demo')
    parser.add_argument('mode', type=str, choices=['train', 'play'])
    parser.add_argument('game', type=str, choices=['tictactoe', 'taxi', 'othello'])
    parser.add_argument('--player', type=str, default='hc', choices=['hc', 'ch', 'cc', 'hh'])
    parser.add_argument('--simulation', default=3, type=str)
    parser.add_argument('--exploration', default=0.9, type=float)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    args.mode = args.mode.lower()
    assert args.game.lower() in GAMES
    game = GAMES[args.game.lower()]
    args.game = game['name']  # The name of the game to load
    args.epochs = game['epochs']  # The number of epochs
    args.checkpoint = game['checkpoint']  # Checkpoint filename
    args.tie = game['tie']  # When game is tied, how to set value. Typically in this case, 0 is used for the tied game

    return args


def main():
    # Register Games
    register_games()

    # Parse Arguments
    args = parse_args()

    # Set Random Seed
    seed = np.random.randint(0, 50000)
    np.random.seed(seed)

    # Get Environment and Set Actions
    env: BaseEnv = gym.make(args.game)
    mcts = load(args.checkpoint)
    # mcts = None
    if args.mode == 'train':
        if mcts is None:
            mcts = MCTS(env, simulation=args.simulation, max_depth=100, exploration_rate=args.exploration)
        mcts.exploration_rate = args.exploration
        mcts.simulation_depth = 100
        mcts.train(epochs=args.epochs, simulation_depth=70, checkpoint=args.checkpoint, seed=args.seed)
    if args.mode == 'play':
        while True:
            mcts.play(mode=args.player)
    print('DONE!!')


if __name__ == '__main__':
    main()
