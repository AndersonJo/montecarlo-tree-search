from gym.envs.registration import register


def register_games():
    register(
        id='othello-v0',
        entry_point='games.othello:Othello',
    )
