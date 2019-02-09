from abc import ABC

from gym import Env


class BaseEnv(Env, ABC):
    def __init__(self):
        super(BaseEnv, self).__init__()
        self.player = None

    def calculate_reward(self, player) -> int:
        """
        Some games require player-based reward like Othello or Chess.
        :param player:
        """
        raise NotImplementedError('calculate_reward not implemented')

    #
    # def calculate_maximum_depth_penalty(self, player) -> int:
    #     raise NotImplementedError('calculate_maximum_depth_penalty not implemented')

    def change_turn(self):
        raise NotImplementedError('change_turn not implemented')

    def copy(self):
        raise NotImplementedError('copy not implemented')

    def get_legal_actions(self) -> list:
        """

        :return: Action or None
            when there is no action to do return None.
        """
        raise NotImplementedError('get_legal_actions not implemented')

    def to_hashed_state(self, player, state) -> str:
        raise NotImplementedError('to_hashed_state not implemented')

    def play_as_human(self):
        """
        :return: action
        """
        raise NotImplementedError('play_as_human not implemented')
