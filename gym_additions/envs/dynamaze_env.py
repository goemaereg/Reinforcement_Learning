import gym
from gym import spaces
import numpy as np
import random

class DynaMazeEnv(gym.Env):
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
        self.moves_str = { # string versions for display
                0: 'U', #up
                1: 'R',   # right
                2: 'D',   # down
                3: 'L',  # left
                4: 'E', # Explore (when using Explore option)
                5: 'E', # Explore (when using Explore option)
                6: 'E', # Explore (when using Explore option)
                }
        self.terminal = (0,self.width-1) # terminal state
        self.obstacles = ( # idk why i used a tuple there but now it's done sry
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
        self.small_reset()
        self.appear_obstacle = 0

    def small_reset(self):
        self.s = self.start
        return self.s

    def reset(self):
        self.__init__()
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



class BlockingMazeEnv(DynaMazeEnv):
    """ Obstacles form a single line but there is a short path; which disappears
    after some steps (agents need to find the new path)"""

    def __init__(self):
        DynaMazeEnv.__init__(self)
        self.t = 0
        self.start = (5,3)
        self.terminal = (0,self.width-1) # terminal state
        self.obstacles = (
            (3,0), # disappears after 1k steps
            (3,1),
            (3,2),
            (3,3),
            (3,4),
            (3,5),
            (3,6),
            (3,7) # (3,8) appears after 1k steps, blocking the short path
        )
        self.appear_obstacle = 30000

    def step(self, action):
        self.t += 1
        if self.t == self.appear_obstacle:
            l = list(self.obstacles)
            l.pop(0)
            l.append((3,8))
            self.obstacles = tuple(l) # keeping tuples to stay consistent

        return super(BlockingMazeEnv, self).step(action)


class ShortcutMazeEnv(DynaMazeEnv):
    """ Obstacles form a single line blocking the shortest path.
    The path opens up after some steps."""

    def __init__(self):
        super(ShortcutMazeEnv, self).__init__()
        self.t = 0
        self.start = (5,3)
        self.terminal = (0,self.width-1) # terminal state
        self.obstacles = (
            (3,1),
            (3,2),
            (3,3),
            (3,4),
            (3,5),
            (3,6),
            (3,7),
            (3,8) # this one disappears after 3k steps, creating the shortcut
        )
        self.appear_obstacle = 30000 # number of steps before obstacle appears

    def step(self, action):
        self.t += 1
        if self.t == self.appear_obstacle:
            l = list(self.obstacles)
            l.pop(-1)
            self.obstacles = tuple(l) # keeping tuples to stay consistent

        return super(ShortcutMazeEnv, self).step(action)


class ShortcutBlockMazeEnv(DynaMazeEnv):
    """ There is a path left and right but the shortest is cut off after 1k."""

    def __init__(self):
        DynaMazeEnv.__init__(self)
        self.t = 0
        self.start = (5,3)
        self.terminal = (0,self.width-1) # terminal state
        self.obstacles = (
            (3,1),
            (3,2),
            (3,3),
            (3,4),
            (3,5),
            (3,6),
            (3,7)
            # (3,8) appears after 1k steps, blocking the short path
        )
        self.appear_obstacle = 30000 # number of steps before obstacle appears

    def step(self, action):
        self.t += 1
        if self.t == self.appear_obstacle:
            l = list(self.obstacles)
            l.append((3,8))
            self.obstacles = tuple(l) # keeping tuples to stay consistent

        return super(ShortcutBlockMazeEnv, self).step(action)
