from abc import ABC

import pygame
from gym.utils import EzPickle
from pygame.locals import *
import numpy as np

from mcts.env import BaseEnv


class TicTacToeBase(BaseEnv, EzPickle):
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    PLAYERS = {1: 'white', 2: 'black'}

    def __init__(self):
        super(TicTacToeBase, self).__init__()
        self.player = 0
        self.board = np.zeros(3, 3)

    def calculate_reward(self, player) -> int:
        pass

    def calculate_maximum_depth_penalty(self, player) -> int:
        pass

    def change_turn(self):
        self.player = self.WHITE if self.player == self.BLACK else self.BLACK

    def copy(self):
        pass

    def get_legal_actions(self) -> list:
        pass

    def next_player(self):
        pass

    def play_as_human(self):
        pass

    def to_hashed_state(self, player, state) -> str:
        pass

    def step(self, action):
        y, x = action
        assert self.board[y, x] == self.EMPTY

        self.board[y, x] = self.player

        self.change_turn()

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class TicTacToe(TicTacToeBase):

    def __init__(self):
        super(TicTacToe, self).__init__()

        # screen: initialized pygame screen
        # background: game board surface
        self.screen: pygame.Surface = None
        self.background: pygame.Surface = None
        self.init_board()

    def init_board(self):
        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode((300, 325))
        pygame.display.set_caption('Tic-Tac-Toe')

        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((250, 250, 250))

        pygame.draw.line(self.background, (0, 0, 0), (100, 0), (100, 300), 2)
        pygame.draw.line(self.background, (0, 0, 0), (200, 0), (200, 300), 2)

        pygame.draw.line(self.background, (0, 0, 0), (0, 100), (300, 100), 2)
        pygame.draw.line(self.background, (0, 0, 0), (0, 200), (300, 200), 2)

    def play_as_human(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        y, x = self.to_board_position(mouse_y, mouse_x)

        if self.board[y, x] in [self.WHITE, self.BLACK]:
            return None

        # draw an X or O
        # drawMove(board, row, col, XO)

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
            message = self.PLAYERS[self.player] + "'s turn"
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
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

    def play(self):

        # main event loop
        is_running = True

        while is_running:
            for event in pygame.event.get():
                if event.type is QUIT:
                    is_running = False

                elif event.type is MOUSEBUTTONDOWN:
                    # the user clicked; place an X or O
                    self.play_as_human()

                # check for a winner
                self.render()

                # update the display
                showBoard(ttt, board)

    def move(self, y, x, player):
        # determine the center of the square
        center_x = (x * 100) + 50
        center_y = (y * 100) + 50

        # draw the appropriate piece
        if player == self.WHITE:
            pygame.draw.circle(self.background, (0, 0, 0), (center_x, center_y), 44, 2)
        else:
            pygame.draw.line(self.background, (0, 0, 0), (center_x - 22, center_y - 22), \
                             (center_x + 22, center_y + 22), 2)
            pygame.draw.line(self.background, (0, 0, 0), (center_x + 22, center_y - 22), \
                             (center_x - 22, center_y + 22), 2)

        # mark the space as used
        self.board[y, x] = player


def gameWon(board):
    # determine if anyone has won the game
    # ---------------------------------------------------------------
    # board : the game board surface

    global grid, winner

    # check for winning rows
    for row in range(0, 3):
        if ((grid[row][0] == grid[row][1] == grid[row][2]) and \
                (grid[row][0] is not None)):
            # this row won
            winner = grid[row][0]
            pygame.draw.line(board, (250, 0, 0), (0, (row + 1) * 100 - 50), \
                             (300, (row + 1) * 100 - 50), 2)
            break

    # check for winning columns
    for col in range(0, 3):
        if (grid[0][col] == grid[1][col] == grid[2][col]) and \
                (grid[0][col] is not None):
            # this column won
            winner = grid[0][col]
            pygame.draw.line(board, (250, 0, 0), ((col + 1) * 100 - 50, 0), \
                             ((col + 1) * 100 - 50, 300), 2)
            break

    # check for diagonal winners
    if (grid[0][0] == grid[1][1] == grid[2][2]) and \
            (grid[0][0] is not None):
        # game won diagonally left to right
        winner = grid[0][0]
        pygame.draw.line(board, (250, 0, 0), (50, 50), (250, 250), 2)

    if (grid[0][2] == grid[1][1] == grid[2][0]) and \
            (grid[0][2] is not None):
        # game won diagonally right to left
        winner = grid[0][2]
        pygame.draw.line(board, (250, 0, 0), (250, 50), (50, 250), 2)
