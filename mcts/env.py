from abc import ABC
from typing import Tuple

from gym import Env


class BaseEnv(Env, ABC):

    def change_turn(self):
        raise NotImplementedError('change_turn not implemented')

    def to_hashed_state(self, player, state) -> str:
        raise NotImplementedError('to_hashed_state not implemented')

    def get_legal_actions(self):
        raise NotImplementedError('get_legal_actions not implemented')
