import logging

from games.othello import Othello
from mcts2.mcts import MCTS, Node

logger = logging.getLogger('mcts')


class MCTSOthello(MCTS):

    def get_environment_name(self):
        return 'othello'

    def on_init(self):
        self.root.info['turn'] = self.env.turn
        turn = self.env.PLAYERS[self.env.turn]
        logger.info(f'start game | turn:{turn}')

    def on_expand(self, env: Othello, new_node: Node, state):
        # the environment is still not applied for the new node.
        # so the env.turn should be reversed
        turn = env.BLACK if env.turn == env.WHITE else env.WHITE
        new_node.info['turn'] = turn

    def finish_simulation_step(self, simulation_step: int, start_env: Othello, simulation_env: Othello,
                               start_node, cur_node, state, reward, done, info, total_score: int) -> int:
        turn = start_env.turn
        reward = reward if turn == start_env.WHITE else (reward[1], reward[0])
        overcome = reward[0] > reward[1]

        turn = start_env.PLAYERS[turn]
        overcome = 1 if overcome else 0
        logger.debug(f'{simulation_step} finish_simulation_step | turn:{turn} | '
                     f'player:{reward[0]} | opponent:{reward[1]} | overcome:{overcome} | total_score:{total_score}')
        return overcome

    def after_simulation_step(self, start_env: Othello, simulation_env: Othello, start_node, cur_node, state, reward,
                              done, info) -> int:
        pass

    def on_reward(self, env, node, reward, visit):
        print(env.PLAYERS[node.info['turn']], env.PLAYERS[env.turn], self.root.info)
        import ipdb
        ipdb.set_trace()
        pass
