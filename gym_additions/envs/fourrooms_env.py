import gym
from gym import spaces
import numpy as np
import random

class FourRoomsEnv(gym.Env):
    """ Small Gridworld environment with 4 rooms.
    Starting up left, goal in lower-right.
    The main challenge is that the reward is sparse (1_goal)
        """
    def __init__(self):
        self.roomsize = 10
        self.height = 2*self.roomsize +1 # +1 is obstacle width
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


class FourRoomsMinEnv(FourRoomsEnv):
    """ FourRoom environment, but two suboptimal goals were added in the middle
    of the upper right and lower left rooms (on the way to the true goal)
    Rewards: 1 for suboptimal, 100 for optimal.
        """
    def __init__(self):
        super(FourRoomsMinEnv, self).__init__()
        half = self.width // 2 # shortcut
        quarter = half // 2 # shortcut
        self.terminals = ((self.height-1,self.width-1), # terminal state
                          (quarter, half+quarter),
                          (half+quarter, quarter)
                          )

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

        done = (self.s in self.terminals)
        if self.s == self.terminals[0]: # optimal goal
            r = 100
        elif self.s in self.terminals[1:]: #suboptimal goals
            r = 1
        else:
            r = 0
        return self.s, r, done, {}


    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[list(zip(*self.obstacles))] = 'X'
        s[list(zip(*self.terminals))] = 'T'
        s[self.start] = 'S'
        s[self.s] = '.'
        print(s)


class FourRoomsGoalEnv(gym.Env):
    """ Goal-oriented version of the 4Rooms
        """
    def __init__(self, **kawrgs):
        self.roomsize = 3 # very small env
        self.height = 2*self.roomsize +1 # +1 is obstacle width
        self.width = self.height
        half = self.width // 2 # door
        quarter = half // 2 # door
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                # input state
                spaces.Discrete(self.height),
                spaces.Discrete(self.width),
                # goal state
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
            ))
        self.moves = { # primitive moves
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }
        horizontal  = [[i, half] for i in range(self.width)]
        vertical    = [[half, i] for i in range(self.height)]
        self.obstacles = horizontal + vertical
        # now opening the 4 passages
        for state in [[quarter,half], [half,quarter], [half,self.height-1-quarter], [self.height-1-quarter,half]]:
            self.obstacles.remove(state)
        self.reset() # choose start and goal

    def reset(self):
        while True:
            self.start = (np.random.randint(0,self.height), np.random.randint(0,self.width))
            if not self.start in self.obstacles:
                break
        self.s = self.start

        while True:
            self.goal = (np.random.randint(0,self.height), np.random.randint(0,self.width))
            if not self.goal in self.obstacles+[self.s]:
                break

        return (*self.s, *self.goal)

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


        done = (self.s == self.goal)
        return (*self.s, *self.goal), int(done), done, {}


    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[list(zip(*self.obstacles))] = 'X'
        s[self.goal] = 'G'
        s[self.start] = 'S'
        s[self.s] = '.'
        print(s)


class FourRoomsGoalBigEnv(FourRoomsGoalEnv):
    """ Goal-oriented version of the 4Rooms; bigger rooms.
        """
    def __init__(self):
        self.roomsize = 10 # bigger env
        self.height = 2*self.roomsize +1 # +1 is obstacle width
        self.width = self.height
        half = self.width // 2 # door
        quarter = half // 2 # door
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                # input state
                spaces.Discrete(self.height),
                spaces.Discrete(self.width),
                # goal state
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
            ))
        self.moves = { # primitive moves
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }
        horizontal  = [[i, half] for i in range(self.width)]
        vertical    = [[half, i] for i in range(self.height)]
        self.obstacles = horizontal + vertical
        # now opening the 4 passages
        for state in [[quarter,half], [half,quarter], [half,self.height-1-quarter], [self.height-1-quarter,half]]:
            self.obstacles.remove(state)
        self.reset() # choose start and goal
