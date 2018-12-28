import gym

from mcts import load, MCTS, Node

mcts, seed = load()
taxi = gym.make('Taxi-v2')
ACTIONS = {'left': 3, 'right': 2, 'up': 1, 'down': 0, 'pickup0': 4, 'dropoff': 5}
REV_ACTIONS = {v: k for k, v in ACTIONS.items()}

taxi.seed(seed)
state = taxi.reset()
mcts.reset()
while True:
    action = mcts.search_action(state, exploration_rate=0)
    state, reward, done, info = taxi.step(action)
    print(taxi.render(), 'action:', REV_ACTIONS[action], 'reward:', reward)

    if done:
        break
    input()
