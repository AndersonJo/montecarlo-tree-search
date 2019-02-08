import numpy as np
import pygame
from gym.utils import EzPickle
from pygame.locals import *

from mcts.env import BaseEnv


class TicTacToeBase(BaseEnv, EzPickle):
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    PLAYERS = {1: 'O', 2: 'X'}

    def __init__(self):
        super(TicTacToeBase, self).__init__()
        self.player = self.WHITE
        self.board = np.zeros((3, 3))

    def calculate_reward(self, player) -> int:
        pass

    def calculate_maximum_depth_penalty(self, player) -> int:
        pass

    def change_turn(self):
        self.player = self.WHITE if self.player == self.BLACK else self.BLACK

    def copy(self):
        pass

    def get_legal_actions(self) -> list:
        return np.column_stack(np.dstack(np.where(self.board == self.EMPTY))).tolist()

    def next_player(self):
        pass

    def play_as_human(self):
        pass

    def to_hashed_state(self, player, state) -> str:
        return str(self.player) + ''.join(self.board.reshape(-1).astype(np.int).astype(np.str).tolist())

    def step(self, action):

        y, x = action
        assert self.board[y, x] == self.EMPTY

        self.board[y, x] = self.player

        self.is_win()

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def is_win(self) -> bool:
        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board[i, :] == self.player).all()
            cols_win = (self.board[:, i] == self.player).all()

            if rows_win or cols_win:
                return True

        diag1_win = (np.diag(self.board) == self.player).all()
        diag2_win = (np.diag(np.fliplr(self.board)) == self.player).all()

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
        self.init_board()

    def init_board(self):
        pygame.init()
        self.init_screen: pygame.Surface = pygame.display.set_mode((300, 325))
        pygame.display.set_caption('Tic-Tac-Toe')

        self.screen = pygame.Surface(self.init_screen.get_size())
        self.screen = self.screen.convert()
        self.screen.fill((250, 250, 250))

        pygame.draw.line(self.screen, (0, 0, 0), (100, 0), (100, 300), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (200, 0), (200, 300), 2)

        pygame.draw.line(self.screen, (0, 0, 0), (0, 100), (300, 100), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 200), (300, 200), 2)

    def play_as_human(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        y, x = self.to_board_position(mouse_y, mouse_x)

        if self.board[y, x] in [self.WHITE, self.BLACK]:
            return None

        # draw an X or O
        # self.move(y, x, self.player)
        state, reward, done, info = self.step((y, x))
        print(f'state:{state} | reward:{reward} | done:{done}')

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
        self.init_screen.blit(self.screen, (0, 0))
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

    def step(self, action, mode='human'):
        super(TicTacToe, self).step(action)
        y, x = action

        # determine the center of the square
        center_x = (x * 100) + 50
        center_y = (y * 100) + 50

        # draw the appropriate piece
        if self.player == self.WHITE:
            pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), 44, 2)
        else:
            pygame.draw.line(self.screen, (0, 0, 0), (center_x - 22, center_y - 22), \
                             (center_x + 22, center_y + 22), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (center_x + 22, center_y - 22), \
                             (center_x - 22, center_y + 22), 2)

        done = is_win = self.is_win()
        hashed_state = self.to_hashed_state(self.player, self.board)
        reward = 0

        if not done:
            done = np.sum(self.board == self.EMPTY) == 0
        if is_win:
            reward = 1

        pygame.display.update()

        self.change_turn()
        return hashed_state, reward, done, {}

    def is_win(self) -> bool:
        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board[i, :] == self.player).all()
            cols_win = (self.board[:, i] == self.player).all()

            if rows_win:
                pygame.draw.line(self.screen, (250, 0, 0), (0, (i + 1) * 100 - 50), (300, (i + 1) * 100 - 50), 2)
                return True
            elif cols_win:
                pygame.draw.line(self.screen, (250, 0, 0), ((i + 1) * 100 - 50, 0), ((i + 1) * 100 - 50, 300), 2)
                return True

        diag1_win = (np.diag(self.board) == self.player).all()
        diag2_win = (np.diag(np.fliplr(self.board)) == self.player).all()

        if diag1_win:
            pygame.draw.line(self.screen, (250, 0, 0), (50, 50), (250, 250), 2)
            return True
        elif diag2_win:
            pygame.draw.line(self.screen, (250, 0, 0), (250, 50), (50, 250), 2)
            return True

        return False


if __name__ == '__main__':
    ttt = TicTacToe()
    ttt.play()
