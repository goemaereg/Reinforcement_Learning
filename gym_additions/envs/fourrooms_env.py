import gym
from gym import spaces
import numpy as np
import random

class FourRoomsEnv(gym.Env):
    """ Small Gridworld environment with 4 rooms.
    Starting up left, goal in lower-right.
        """
    def __init__(self):
        roomsize = 25
        self.height = 2*roomsize +1 # +1 is obstacle width
        self.width = self.height
        half = self.width // 2 # shortcut
        quarter = half // 2 # shortcut
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
        self.terminal = (self.height-1,self.width-1) # terminal state
        horizontal  = [[i, half] for i in range(self.width)]
        vertical    = [[half, i] for i in range(self.height)]
        self.obstacles = horizontal + vertical
        # now opening the 4 passages
        for state in [[quarter,half], [half,quarter], [half,self.height-quarter], [self.height-quarter,half]]:
            self.obstacles.remove(state)
        self.start = (0,0)
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
