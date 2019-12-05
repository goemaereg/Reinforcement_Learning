import gym
from gym import spaces
from gym.envs.classic_control import AcrobotEnv
from utils import TileCoder, one_hot
import numpy as np
import itertools

class AcrobotTileEnv(AcrobotEnv):
    """ Copy of Acrobot with Tiles as state feature representation.
    The state is now a one_hot vector of the tiles it touches."""

    def __init__(self):
        super(AcrobotTileEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [3]*n_dim
        lims = np.array([self.observation_space.low, self.observation_space.high]).T
        tilings = n_dim
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Discrete(self.T.n_tiles)

    def step(self, action):
        s,r,d,i = super(AcrobotTileEnv, self).step(action)
        return self.T[s],r,d,i

    def reset(self):
        s = super(AcrobotTileEnv, self).reset()
        return self.T[s]


class AcrobotPolyEnv(AcrobotEnv):
    """ States are polynomials (1,x_1..6,and products) of Acrobot 6 inputs."""

    def __init__(self):
        super(AcrobotPolyEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = [(i,j) for i in range(n_dim) for j in range(n_dim) if i!= j]
        self.observation_space = spaces.Discrete(1+n_dim+len(self.prod_indices))

    def _converter(self, s):
        """ Converts the 4-element state into the polynomial one."""
        double_products = [s[a]*s[b] for a,b in self.prod_indices]
        return np.array([1] + list(s) + double_products)

    def step(self, action):
        s,r,d,i = super(AcrobotPolyEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(AcrobotPolyEnv, self).reset()
        return self._converter(s)

class AcrobotDoubleEnv(AcrobotEnv):
    """ Combination of both Poly and Tile for ActorCritics."""

    def __init__(self):
        super(AcrobotDoubleEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = [(i,j) for i in range(n_dim) for j in range(n_dim) if i!= j]
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [10]*6
        lims = np.array([self.observation_space.low, self.observation_space.high]).T
        tilings = 10
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
        s,r,d,i = super(AcrobotDoubleEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(AcrobotDoubleEnv, self).reset()
        return self._converter(s)
