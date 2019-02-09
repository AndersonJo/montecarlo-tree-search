from abc import ABC
from copy import deepcopy
from itertools import product
from typing import Tuple, Union

import numpy as np
import pygame
from gym.utils import EzPickle
from pygame.locals import *

from mcts.env import BaseEnv


class TicTacToeBase(BaseEnv, EzPickle):
    EMPTY = 0
    WHITE = 1  # O
    BLACK = 2  # X
    PLAYERS = {1: 'O', 2: 'X'}

    def __init__(self):
        super(TicTacToeBase, self).__init__()

        self.player = self.WHITE
        self.board = np.zeros((3, 3), dtype=np.int)
        self._cache = {}

    def change_turn(self):
        self.player = self.WHITE if self.player == self.BLACK else self.BLACK

    def copy(self):
        copied = deepcopy(self)
        copied.player = self.player
        copied.board = self.board.copy()
        return copied

    def calculate_reward(self, player) -> int:
        is_win = self.is_win(player)
        if is_win:
            return 1
        return 0

    def get_legal_actions(self) -> list:
        legal_actions = np.column_stack(np.dstack(np.where(self.board == self.EMPTY)))
        legal_actions = list(map(lambda x: tuple(x), legal_actions))

        return legal_actions

    def to_hashed_state(self, player, state) -> str:
        return str(self.player) + ''.join(self.board.reshape(-1).astype(np.int).astype(np.str).tolist())

    def step(self, action):
        y, x = action
        assert self.board[y, x] == self.EMPTY

        self.board[y, x] = self.player

        done = is_win = self.is_win(self.player)
        hashed_state = self.to_hashed_state(self.player, self.board)
        reward = 0

        if not done:
            done = np.sum(self.board == self.EMPTY) == 0
        if is_win:
            reward = 1

        self.change_turn()
        return hashed_state, reward, done, {}

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        self.player = np.random.choice([self.WHITE, self.BLACK])
        return self.to_hashed_state(self.player, self.board)

    def is_win(self, player) -> bool:
        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board[i, :] == player).all()
            cols_win = (self.board[:, i] == player).all()

            if rows_win or cols_win:
                return True

        diag1_win = (np.diag(self.board) == player).all()
        diag2_win = (np.diag(np.fliplr(self.board)) == player).all()

        if diag1_win or diag2_win:
            return True
        return False


class TicTacToe(TicTacToeBase):

    def __init__(self):
        super(TicTacToe, self).__init__()

        # screen: initialized pygame screen
        # background: game board surface
        self.init_screen: pygame.Surface = None
        self.screen: pygame.Surface = None

        pygame.init()
        self.init_screen: pygame.Surface = pygame.display.set_mode((300, 325))
        pygame.display.set_caption('Tic-Tac-Toe')

        self.screen = pygame.Surface(self.init_screen.get_size())
        self.screen = self.screen.convert()
        self.init_board()

    def init_board(self):
        self.screen.fill((250, 250, 250))

        pygame.draw.line(self.screen, (0, 0, 0), (100, 0), (100, 300), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (200, 0), (200, 300), 2)

        pygame.draw.line(self.screen, (0, 0, 0), (0, 100), (300, 100), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 200), (300, 200), 2)

    def copy(self):
        env = super(TicTacToe, self).copy()
        env.screen = self.screen
        env.init_screen = self.init_screen
        env.screen.fill((250, 250, 250))
        return env

    def play_as_human(self) -> Union[Tuple[int, int], None]:
        action = None
        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type is QUIT:
                    is_running = False
                    break

                elif event.type is MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    y, x = self.to_board_position(mouse_y, mouse_x)

                    if self.board[y, x] in [self.WHITE, self.BLACK]:
                        continue
                    else:
                        action = (y, x)
                        is_running = False
                        break

        return action

    def to_board_position(self, mouse_y, mouse_x):
        if mouse_y < 100:
            y = 0
        elif mouse_y < 200:
            y = 1
        else:
            y = 2

        # determine the column the user clicked
        if mouse_x < 100:
            x = 0
        elif mouse_x < 200:
            x = 1
        else:
            x = 2

        # return the tuple containg the row & column
        return (y, x)

    def _render_status(self):
        winner = None
        if (winner is None):
            message = f'Turn: {self.PLAYERS[self.player]}'
        else:
            message = winner + " won!"

        # render the status message
        font = pygame.font.Font(None, 24)
        text = font.render(message, 1, (10, 10, 10))

        # copy the rendered message onto the board
        self.screen.fill((250, 250, 250), (0, 300, 300, 25))
        self.screen.blit(text, (10, 300))

    def render(self, mode='human'):
        self._render_status()

        self.screen.fill((250, 250, 250))

        pygame.draw.line(self.screen, (0, 0, 0), (100, 0), (100, 300), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (200, 0), (200, 300), 2)

        pygame.draw.line(self.screen, (0, 0, 0), (0, 100), (300, 100), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 200), (300, 200), 2)

        for y, x in product(range(3), range(3)):
            if self.board[y, x] not in self.PLAYERS:
                continue

            # determine the center of the square
            center_x = (x * 100) + 50
            center_y = (y * 100) + 50
            player = int(self.board[y, x])
            # draw the appropriate piece
            if player == self.WHITE:
                pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), 44, 2)
            else:
                pygame.draw.line(self.screen, (0, 0, 0), (center_x - 22, center_y - 22), \
                                 (center_x + 22, center_y + 22), 2)
                pygame.draw.line(self.screen, (0, 0, 0), (center_x + 22, center_y - 22), \
                                 (center_x - 22, center_y + 22), 2)

        self._draw_line(self.player)
        self.init_screen.blit(self.screen, (0, 0))
        pygame.display.update()

    def _draw_line(self, player) -> bool:
        for player in [self.WHITE, self.BLACK]:
            for i in range(0, 3):
                # Checks rows and columns for match
                rows_win = (self.board[i, :] == player).all()
                cols_win = (self.board[:, i] == player).all()

                if rows_win:
                    pygame.draw.line(self.screen, (250, 0, 0), (0, (i + 1) * 100 - 50), (300, (i + 1) * 100 - 50), 2)
                    return True
                elif cols_win:
                    pygame.draw.line(self.screen, (250, 0, 0), ((i + 1) * 100 - 50, 0), ((i + 1) * 100 - 50, 300), 2)
                    return True

            diag1_win = (np.diag(self.board) == player).all()
            diag2_win = (np.diag(np.fliplr(self.board)) == player).all()

            if diag1_win:
                pygame.draw.line(self.screen, (250, 0, 0), (50, 50), (250, 250), 2)
                return True
            elif diag2_win:
                pygame.draw.line(self.screen, (250, 0, 0), (250, 50), (50, 250), 2)
                return True

        return False

    def reset(self):
        hashed_state = super(TicTacToe, self).reset()
        self.init_board()

        return hashed_state

    def play(self):

        # main event loop
        is_running = True
        self.render()
        while True:
            action = self.play_as_human()

            if action is None:
                break

            state, reward, done, info = self.step(action)
            self.render()

            print(self.board)
            print(f'state:{state} | reward:{reward} | done:{done}')
            if done:
                break


if __name__ == '__main__':
    ttt = TicTacToe()
    ttt.play()
    ttt.reset()
