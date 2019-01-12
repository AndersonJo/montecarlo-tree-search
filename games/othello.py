from itertools import product
from typing import List, Tuple, Union

import numpy as np
import pygame
from gym import Env, spaces


class OthelloBase(Env):
    EMPTY = 0
    HINT = 1
    WHITE = 2
    BLACK = 3

    END_GAME = 0
    STEP_DONE = 1  # put the piece on the board
    STEP_NOPE = -1  # no place to put the piece

    def __init__(self, board_width=8, board_height=8):
        assert board_width % 2 == 0, 'Board width should be an even number'
        assert board_height % 2 == 0, 'Board height should be an even number'
        assert board_width >= 4, 'Board width should be more than 4'
        assert board_height >= 4, 'Board height should be more than 4'

        self.action_space = spaces.MultiDiscrete((board_height, board_width))

        self.board_width = board_width
        self.board_height = board_height
        self.board: np.ndarray = None
        self.turn: int = self.WHITE

        self.reset()

    def reset(self):
        if self.board is None:
            self.board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        else:
            self.board[:] = self.EMPTY

        x = self.board_width // 2
        y = self.board_height // 2

        self.board[y, x] = self.WHITE
        self.board[y - 1, x - 1] = self.WHITE
        self.board[y, x - 1] = self.BLACK
        self.board[y - 1, x] = self.BLACK

    def is_valid_position(self, x, y, player) -> Union[List[Tuple[int, int]], None]:
        """
        :param x: the index position of x
        :param y: the index position of y
        :param player: white or black (1, 2)
        :return a list of flippable positions where opponent's pieces are
        """
        x_position, y_position = x, y
        if player == self.WHITE:
            opponent = self.BLACK
        else:
            opponent = self.WHITE

        if (self.board[y, x] != self.EMPTY and self.board[y, x] != self.HINT) or not self.is_on_board(x, y):
            return None

        # Temporarily set the position as the player's one.
        self.board[y_position, x_position] = player

        # Find flippable positions on which opponent's pieces
        flips = list()
        for dx, dy in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = x_position, y_position
            x += dx
            y += dy
            if self.is_on_board(x, y) and self.board[y, x] == opponent:
                x += dx
                y += dy

                if not self.is_on_board(x, y):
                    continue

                while self.board[y, x] == opponent:
                    x += dx
                    y += dy
                    if not self.is_on_board(x, y):
                        break

                if not self.is_on_board(x, y):
                    continue

                if self.board[y, x] == player:
                    while True:
                        x -= dx
                        y -= dy
                        if x == x_position and y == y_position:
                            break
                        flips.append((x, y))

        # Turn back the position as empty space
        self.board[y_position, x_position] = self.EMPTY

        if not flips:
            return None
        return flips

    def is_on_board(self, x, y):
        return 0 <= x < self.board_width and 0 <= y < self.board_height

    def is_end(self):
        return not bool(np.sum(self.board == self.EMPTY))

    def calculate_score(self):
        # Determine the score by counting the tiles.
        white_score = 0
        black_score = 0
        for x, y in product(range(self.board_width), range(self.board_height)):
            if self.board[y, x] == self.WHITE:
                white_score += 1
            if self.board[y, x] == self.BLACK:
                black_score += 1
        return white_score, black_score

    def get_valid_positions(self):
        valid_positions = []
        for x, y in product(range(self.board_width), range(self.board_height)):
            if self.is_valid_position(x, y, self.turn):
                valid_positions.append((x, y))
        return valid_positions

    def render(self, mode='human'):
        pass

    def step(self, action: Tuple[int, int]):
        """
        :param action: a tuple of x and y coordinate
        :return: game status
        """
        x, y = action

        flip_positions = self.is_valid_position(x, y, self.turn)
        if flip_positions is None:
            return self.STEP_NOPE

        self.board[y, x] = self.turn

        for x, y in flip_positions:
            self.board[y, x] = self.turn

        if self.is_end():
            return self.END_GAME

        self.change_turn()

        return self.STEP_DONE

    def change_turn(self):
        self.turn = self.WHITE if self.turn == self.BLACK else self.BLACK

    def hint(self):
        self.remove_hint()

        valid_positions = self.get_valid_positions()
        for x, y in valid_positions:
            self.board[y, x] = self.HINT

    def remove_hint(self):
        self.board[self.board == self.HINT] = self.EMPTY


class Othello(OthelloBase):
    GAME_WIDTH = 500
    GAME_HEIGHT = 500
    SPACE_PIXEL = 50
    GRID_COLOR = (30, 30, 30)
    BACKGROUND_COLOR = (35, 124, 51)
    WHITE_COLOR = (230, 230, 230)
    BLACK_COLOR = (30, 30, 30)
    HINT_COLOR = (150, 150, 90)
    TEXT_COLOR = (0, 0, 0)
    FPS = 10

    def __init__(self, board_width=8, board_height=8, show_hint=True):
        super(Othello, self).__init__(board_width, board_height)

        self.show_hint = show_hint
        self.X_MARGIN = int((self.GAME_WIDTH - (self.board_width * self.SPACE_PIXEL)) / 2)
        self.Y_MARGIN = int((self.GAME_HEIGHT - (self.board_height * self.SPACE_PIXEL)) / 2)

        # Initialize PyGame
        pygame.init()
        pygame.display.set_caption('Reversi - Anderson')
        self.clock = pygame.time.Clock()
        self.screen: pygame.Surface = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT))
        self.screen.fill(self.BACKGROUND_COLOR)

        self.font = pygame.font.Font('freesansbold.ttf', 16)
        self.bigfont = pygame.font.Font('freesansbold.ttf', 32)

        pygame.display.update()

    def render(self, mode='human'):

        self.screen.fill(self.BACKGROUND_COLOR)

        # Fill hints
        if self.show_hint:
            self.hint()

        # Draw grid lines of the board.
        for x in range(self.board_width + 1):
            # Draw the horizontal lines.
            startx = (x * self.SPACE_PIXEL) + self.X_MARGIN
            starty = self.Y_MARGIN
            endx = (x * self.SPACE_PIXEL) + self.X_MARGIN
            endy = self.Y_MARGIN + (self.board_height * self.SPACE_PIXEL)
            pygame.draw.line(self.screen, self.GRID_COLOR, (startx, starty), (endx, endy), 2)

        for y in range(self.board_height + 1):
            # Draw the vertical lines.
            startx = self.X_MARGIN
            starty = (y * self.SPACE_PIXEL) + self.Y_MARGIN
            endx = self.X_MARGIN + (self.board_width * self.SPACE_PIXEL)
            endy = (y * self.SPACE_PIXEL) + self.Y_MARGIN
            pygame.draw.line(self.screen, self.GRID_COLOR, (startx, starty), (endx, endy), 2)

        # Draw the black & white tiles or hint spots.
        for y in range(self.board_height):
            for x in range(self.board_width):
                centerx, centery = self._pixel_to_coord(x, y)
                if self.board[y, x] == self.WHITE or self.board[y, x] == self.BLACK:
                    if self.board[y, x] == self.WHITE:
                        color = self.WHITE_COLOR
                    else:
                        color = self.BLACK_COLOR
                    pygame.draw.circle(self.screen, color, (centerx, centery), int(self.SPACE_PIXEL / 2) - 4)
                if self.board[y, x] == self.HINT:
                    pygame.draw.circle(self.screen, self.HINT_COLOR,
                                       (centerx, centery), int(self.SPACE_PIXEL / 4) - 4)

        pygame.display.update()

    def render_info(self):
        # Draws scores and whose turn it is at the bottom of the screen.
        white_score, black_score = self.calculate_score()

        if self.is_end():
            status = 'End of the Game. Press ESC'
        else:
            status = 'White Turn' if self.turn == self.WHITE else 'Black Turn'

        text = 'White:{0}   Black:{1}   {2}'.format(white_score, black_score, status)
        score_surface = self.font.render(text, True, self.TEXT_COLOR)
        scoreRect = score_surface.get_rect()

        scoreRect.bottomleft = (self.SPACE_PIXEL, self.GAME_HEIGHT - 10)

        self.screen.blit(score_surface, scoreRect)

        pygame.display.update()

    def _pixel_to_coord(self, x, y):
        center_x = self.X_MARGIN + x * self.SPACE_PIXEL + int(self.SPACE_PIXEL / 2)
        center_y = self.Y_MARGIN + y * self.SPACE_PIXEL + int(self.SPACE_PIXEL / 2)
        return center_x, center_y

    def _get_clicked_space(self, mousex, mousey):
        # Return a tuple of two integers of the board space coordinates where
        # the mouse was clicked. (Or returns None not in any space.)
        for x in range(self.board_width):
            for y in range(self.board_height):

                if (mousex > x * self.SPACE_PIXEL + self.X_MARGIN) and \
                        (mousex < (x + 1) * self.SPACE_PIXEL + self.X_MARGIN) and \
                        (mousey > y * self.SPACE_PIXEL + self.Y_MARGIN) and \
                        (mousey < (y + 1) * self.SPACE_PIXEL + self.Y_MARGIN):
                    return x, y
        return None

    def play(self, random_play=False):
        def _is_quit_game(event):
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return True
            return False

        self.turn = self.WHITE
        self.render()

        while True:

            valid_positions = self.get_valid_positions()
            if not valid_positions:
                # No place to put the piece.
                # End of the Game
                self.change_turn()
                valid_positions = self.get_valid_positions()
                if not valid_positions:
                    break

            space_xy = None
            if random_play:
                rand_idx = np.random.randint(0, len(valid_positions))
                space_xy = valid_positions[rand_idx]
            else:
                while space_xy is None:
                    for event in pygame.event.get():  # event handling loop
                        if _is_quit_game(event):  # Quit the game?
                            return

                        if event.type == pygame.MOUSEBUTTONUP:
                            mouse_x, mouse_y = event.pos
                            # Convert pixel coordinate to board coordinate
                            space_xy = self._get_clicked_space(mouse_x, mouse_y)

                            if space_xy is not None and not self.is_valid_position(space_xy[0], space_xy[1], self.turn):
                                space_xy = None

            status = self.step((space_xy[0], space_xy[1]))

            # Render the game board
            self.render()
            self.render_info()

            if status == self.END_GAME:
                while True:
                    for event in pygame.event.get():  # event handling loop
                        if _is_quit_game(event):
                            return
                        elif event.type == pygame.MOUSEBUTTONUP:
                            return

            # Foward
            self.clock.tick(self.FPS)

        white_score, black_score = self.calculate_score()


def dev():
    reversi = Othello()
    reversi.play(random_play=True)


if __name__ == '__main__':
    dev()
