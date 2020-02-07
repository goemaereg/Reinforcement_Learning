import gym
from gym import spaces
import numpy as np
import random

class LocalMinEnv(gym.Env):
    """ Small vertical Gridworld environment, with a suboptimal terminal state
    on the way to the true objective.
    Starting up, suboptimal goal midway, goal down.
    Reward 1 for the suboptimal goal, 10 for optimal.
        """
    def __init__(self):
        self.height = 11
        self.width = 3
        half = self.height // 2 # suboptimal goal position
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
        self.moves_str = {
                0: 'U', #up
                1: 'R',   # right
                2: 'D',   # down
                3: 'L',  # left
                4: 'E', # Explore (when using Explore option)
        }
        self.terminals = [(half, 1), (self.height-1, 1)] # terminal state
        self.start = (0,1)
        # begin in start state
        self.reset()

    def reset(self):
        self.s = self.start
        return self.s

    def step(self, action):
        """ Moves the agent in the action direction."""
        # Next, moving according to action
        x, y = self.moves[action]
        self.s = self.s[0] + x, self.s[1] + y

        # Finally, setting the agent back into the grid if fallen out
        self.s = (max(0, self.s[0]), max(0, self.s[1]))
        self.s = (min(self.s[0], self.height - 1),
                  min(self.s[1], self.width - 1))

        if self.s == self.terminals[0]: # suboptimal
            r = 1
        elif self.s == self.terminals[1]:
            r = 10
        else:
            r = 0
        return self.s, r, self.s in self.terminals, {}


    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[list(zip(*self.terminals))] = 'T'
        s[self.start] = 'S'
        s[self.s] = '.'
        print(s)
