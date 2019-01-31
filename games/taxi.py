from gym.utils import EzPickle

from mcts.env import BaseEnv


class Taxi(BaseEnv, EzPickle):
    def __init__(self):
        super(Taxi, self).__init__()


    def calculate_reward(self, player) -> int:
        pass

    def calculate_maximum_depth_penalty(self, player) -> int:
        pass

    def change_turn(self):
        pass

    def copy(self):
        pass

    def get_legal_actions(self) -> list:
        pass

    def next_player(self):
        pass

    def to_hashed_state(self, player, state) -> str:
        pass

    def play_as_human(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
