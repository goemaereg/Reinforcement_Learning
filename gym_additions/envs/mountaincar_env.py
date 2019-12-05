import gym
from gym import spaces
from gym.envs.classic_control import MountainCarEnv
from utils import TileCoder, one_hot
import numpy as np
import itertools

class MountainCarTileEnv(MountainCarEnv):
    """ Copy of MountainCar with Tiles as state feature representation.
    The state is now a one_hot vector of the tiles it touches."""

    def __init__(self):
        super(MountainCarTileEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [5]*n_dim
        lims = np.array([self.observation_space.low, self.observation_space.high]).T
        tilings = 5
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Discrete(self.T.n_tiles)

    def step(self, action):
        s,r,d,i = super(MountainCarTileEnv, self).step(action)
        tile_s = one_hot(self.T[s], self.T.n_tiles)
        return tile_s,r,d,i

    def reset(self):
        s = super(MountainCarTileEnv, self).reset()
        tile_s = one_hot(self.T[s], self.T.n_tiles)
        return tile_s

class MountainCarPolyEnv(MountainCarEnv):
    """ States are polynomials (1,x_1..6,and products) of MountainCar inputs."""

    def __init__(self):
        super(MountainCarPolyEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = [(i,j) for i in range(n_dim) for j in range(n_dim)]
        self.observation_space = spaces.Discrete(1+n_dim+len(self.prod_indices))

    def _converter(self, s):
        """ Converts the 4-element state into the polynomial one."""
        double_products = [s[a]*s[b] for a,b in self.prod_indices]
        return np.array([1] + list(s) + double_products)

    def step(self, action):
        s,r,d,i = super(MountainCarPolyEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(MountainCarPolyEnv, self).reset()
        return self._converter(s)

class MountainCarDoubleEnv(MountainCarEnv):
    """ Both Poly and Tile for ActorCritics """

    def __init__(self):
        super(MountainCarDoubleEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = [(i,j) for i in range(n_dim) for j in range(n_dim)]
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [5]*n_dim
        lims = np.array([self.observation_space.low, self.observation_space.high]).T
        tilings = 5
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1+n_dim+len(self.prod_indices)),
            spaces.Discrete(self.T.n_tiles)
        ))

    def _converter(self, s):
        """ Converts the 4-element state into the polynomial one."""
        double_products = [s[a]*s[b] for a,b in self.prod_indices]
        return (np.array([1] + list(s) + double_products), self.T[s])

    def step(self, action):
        s,r,d,i = super(MountainCarDoubleEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(MountainCarDoubleEnv, self).reset()
        return self._converter(s)
