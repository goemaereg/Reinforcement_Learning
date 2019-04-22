from gym.envs.registration import register

register(
        id='Gridworld-v0',
        entry_point='gym_additions.envs:GridworldEnv',
        )
register(
        id='WindyGridworld-v0',
        entry_point='gym_additions.envs:WindyGridworldEnv',
        )
register(
        id='Cliff-v0',
        entry_point='gym_additions.envs:CliffEnv',
        )
register(
        id='TicTacToe-v0',
        entry_point='gym_additions.envs:TicTacToeEnv',
        )
