import argparse
from argparse import Namespace

import gym
import numpy as np
from tqdm import tqdm

from games import register_games
from games.othello import Othello
from mcts.mcts import MCTS

ACTIONS = {'left': 3, 'right': 2, 'up': 1, 'down': 0, 'pickup': 4, 'dropoff': 5}

GAMES = {'taxi': 'Taxi-v2',
         'othello': 'othello-v0'}


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


def train(game, epochs, simulation, seed, checkpoint=None):
    env: Othello = gym.make(game)
    print('sssssssss')
    print(env.get_valid_positions())
    return
    if checkpoint is not None:
        pass
        # mcts = load(checkpoint)
    else:
        mcts = MCTS(actions=list(ACTIONS.values()))

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
    # mcts, seed = load(args.checkpoint)
    # taxi = gym.make('Taxi-v2')
    # demo(taxi, mcts, seed=seed)


def main():
    # Register Games
    register_games()

    # Parse Arguments
    args = parse_args()

    # Set Random Seed
    seed = np.random.randint(0, 50000)
    np.random.seed(seed)
    seed = 38260
    train(args.game, epochs=30000000, simulation=15000000, seed=seed)


if __name__ == '__main__':
    main()
