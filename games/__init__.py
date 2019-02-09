from gym.envs.registration import register


def register_games():
    register(
        id='Othello-v0',
        entry_point='games.othello:Othello',
    )

    register(
        id='TicTacToe-v0',
        entry_point='games.tictactoe:TicTacToe'
    )
