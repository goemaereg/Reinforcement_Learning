import gym
from gym import spaces
from gym.envs.classic_control import AcrobotEnv
from utils import TileCoder, one_hot
import numpy as np

class AcrobotTileEnv(AcrobotEnv):
    """ Copy of Acrobot with Tiles as state feature representation.
    The state is now a one_hot vector of the tiles it touches."""

    def __init__(self):
        super(AcrobotTileEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [10]*6
        lims = np.array([self.observation_space.low, self.observation_space.high]).T
        tilings = 8
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Discrete(self.T.n_tiles)

    def step(self, action):
        s,r,d,i = super(AcrobotTileEnv, self).step(action)
        return self.T[s],r,d,i

    def reset(self):
        s = super(AcrobotTileEnv, self).reset()
        return self.T[s]
