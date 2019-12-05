import gym
from gym import spaces
import numpy as np

class ShortCorridorEnv(gym.Env):
    """ 4 states in a line, but the three first have the same feature vector.
    The last state (3) is termination. We start at state 0 and have right/left
    actions options. However, state 1 reverses the actions' consequences.
    The optimal policy is therefore stochastic with p(right) ~= 0.59
    Reward is -1 for all transitions
        """
    def __init__(self):
        self.action_space = spaces.Discrete(2) # left & right
        self.observation_space = spaces.Discrete(2) # {0 1 2} and 3
        self.moves = {
            0: -1,  # left
            1: 1,   # right
        }
        self.state_to_obs = {
            0: np.array([0,1]),
            1: np.array([0,1]),
            2: np.array([0,1]),
            3: np.array([1,0]),
        }
        self.start = 0
        self.terminal = 3
        # begin in start state
        self.reset()

    def reset(self):
        self.s = self.start
        return self.state_to_obs[self.s]

    def step(self, action):
        """ Moves the agent in the action direction.
        The state 1 reverses the commands. """
        move = self.moves[action]
        if self.s == 1: # reverse command
            move *= -1
        self.s += move
        # setting the agent back into the grid
        self.s = np.clip(self.s, 0, 3)

        # Returns; anything brings a reward of -1
        done = (self.s == self.terminal)
        return self.state_to_obs[self.s], -1, done, {}

    def render(self):
        visu = np.zeros(self.terminal+1, dtype=int).astype(str)
        visu[self.start] = 'S'
        visu[self.terminal] = 'T'
        visu[self.s] = '.'
        print(str(visu))
