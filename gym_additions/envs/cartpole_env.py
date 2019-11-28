import gym
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from utils import TileCoder, one_hot
import numpy as np

class CartPoleTileEnv(CartPoleEnv):
    """ Copy of CartPole with Tiles as state feature representation.
    The state is now a one_hot vector of the tiles it touches."""

    def __init__(self):
        super(CartPoleTileEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [5]*n_dim
        clipper = 3
        lims = np.array([self.observation_space.low, self.observation_space.high]).T.clip(-clipper,clipper)
        tilings = 8
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Discrete(self.T.n_tiles)

    def step(self, action):
        s,r,d,i = super(CartPoleTileEnv, self).step(action)
        tile_s = one_hot(self.T[s], self.T.n_tiles)
        return tile_s,r,d,i

    def reset(self):
        s = super(CartPoleTileEnv, self).reset()
        tile_s = one_hot(self.T[s], self.T.n_tiles)
        return tile_s
