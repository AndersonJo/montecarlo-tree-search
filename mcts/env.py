from abc import ABC

from gym import Env


class BaseEnv(Env, ABC):
    def __init__(self):
        super(BaseEnv, self).__init__()
        self.init_player = 1  # Human player or player who will get reward for the game
        self.cur_player = 1  # Current player of the game

    def best_action(self):
        pass

    def copy(self):
        raise NotImplementedError('copy not implemented')

    def get_legal_actions(self) -> list:
        """
        If player 1 has no option to choose an action, change the turn.
        After changing turn, there is still no action to do then return None

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
