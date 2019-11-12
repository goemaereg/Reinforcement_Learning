import gym
from gym import spaces
import numpy as np

class WindyGridworldEnv(gym.Env):
    """ Small gridworld with upwards varying wind."""
    def __init__(self):
        self.height = 7
        self.width = 10
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }
        self.terminal = (3,7) # arbitrary terminal state

        # begin in start state
        self.reset()

    def step(self, action):
        """ Moves the agent in the action direction + upwards wind.
            Note that the wind depends on the position, can be 0, 1, 2"""
        # First applying the wind
        if self.S[1] in (3, 4, 5, 8):
            self.S = self.S[0] - 1, self.S[1]
        elif self.S[1] in (6, 7):
            self.S = self.S[0] - 2, self.S[1]

        # Next, moving according to action
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # Finally, setting the agent back into the grid if fallen out
        self.S = (max(0, self.S[0]), max(0, self.S[1]))
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        # Returns; anything brings a reward of -1
        return self.S, -1, self.S == self.terminal, {}

    def reset(self):
        """ This version has a fixed inital state """
        self.S = (3, 0)
        return self.S

    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[self.terminal] = 'T'
        s[self.S] = '.'
        print(s)

class WindyGridworldKingEnv(WindyGridworldEnv):
    """ Windy gridworld with chess King's moves."""
    def __init__(self):
        self.height = 7
        self.width = 10
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                0: (-1, 0),  # up
                1: (-1, 1),  # up right
                2: (0, 1),   # right
                3: (1, 1),   # down right
                4: (1, 0),   # down
                5: (1, -1),  # down left
                6: (0, -1),  # left
                7: (-1, -1), # left up
                }
        self.terminal = (3,7) # arbitrary terminal state

        # begin in start state
        self.reset()


class WindyGridworldStochaEnv(WindyGridworldKingEnv):
    """ Windy gridworld with chess King's moves and stochastic wind."""
    def step(self, action):
        """ Moves the agent in the action direction + upwards stocha wind.
            Note that the wind depends on the position, can be 0, 1, 2"""
        # First applying the wind
        stocha = np.random.randint(3)-1 # stochastic component

        if self.S[1] in (3, 4, 5, 8):
            self.S = self.S[0] - 1 - stocha, self.S[1]
        elif self.S[1] in (6, 7):
            self.S = self.S[0] - 2 - stocha, self.S[1]

        # Next, moving according to action
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # Finally, setting the agent back into the grid if fallen out
        self.S = (max(0, self.S[0]), max(0, self.S[1]))
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        # Returns; anything brings a reward of -1
        return self.S, -1, self.S == self.terminal, {}
