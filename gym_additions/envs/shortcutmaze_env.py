import gym
from gym import spaces
import numpy as np
import random

class ShortcutMazeEnv(gym.Env):
    """ Small Gridworld environment as a 6x9 grid with obstacles
        """
    def __init__(self):
        self.height = 6
        self.width = 9
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
        self.terminal = (0,self.width-1) # terminal state
        self.obstacles = (
            (1,2),
            (2,2),
            (3,2),
            (4,5),
            (0,self.width-2),
            (1,self.width-2),
            (2,self.width-2)
        )
        self.start = (2,0)
        # begin in start state
        self.reset()

    def reset(self):
        self.s = self.start
        return self.s

    def step(self, action):
        """ Moves the agent in the action direction."""
        # Next, moving according to action
        x, y = self.moves[action]
        if (self.s[0]+x, self.s[1]+y) not in self.obstacles:
            # move is allowed
            self.s = self.s[0] + x, self.s[1] + y

            # Finally, setting the agent back into the grid if fallen out
            self.s = (max(0, self.s[0]), max(0, self.s[1]))
            self.s = (min(self.s[0], self.height - 1),
                      min(self.s[1], self.width - 1))

        done = (self.s == self.terminal)
        return self.s, int(done), done, {}


    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[list(zip(*self.obstacles))] = 'X'
        s[self.terminal] = 'T'
        s[self.start] = 'S'
        s[self.s] = '.'
        print(s)
