import gym
from gym import spaces
import numpy


class GridworldEnv(gym.Env):
    """ Small Gridworld environment as a 4by4 grid with 2 terminal states.
        """
    def __init__(self):
        self.height = 4
        self.width = 4
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
        self.terminals = ((0,0), (3,3)) # arbitrary terminal states

        # begin in start state
        self.reset()

    def step(self, action):
        """ Moves the agent in the action direction.
            """
        # Next, moving according to action
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # Finally, setting the agent back into the grid if fallen out
        self.S = (max(0, self.S[0]), max(0, self.S[1]))
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        # Returns; anything brings a reward of -1
        return self.S, -1, self.S in self.terminals, {}

    def reset(self):
        self.S = (3, 0)
        return self.S

    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[self.terminals[0]] = s[self.terminals[1]] = 'T'
        s[self.S] = '.'
        print(s)
