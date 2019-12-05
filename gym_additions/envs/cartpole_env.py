import gym
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from utils import TileCoder, one_hot
import numpy as np
import itertools

class CartPoleTileEnv(CartPoleEnv):
    """ Copy of CartPole with Tiles as state feature representation.
    The state is now a one_hot vector of the tiles it touches."""

    def __init__(self):
        super(CartPoleTileEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [5]*n_dim
        lims = [[-2,2], [-3,3], [-.5, .5], [-4,4]]
#        lims = np.array([self.observation_space.low, self.observation_space.high]).T.clip(-self.clipper,self.clipper)
        tilings = n_dim
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Discrete(self.T.n_tiles)

    def _encoder(self, s):
        return one_hot(self.T[s], self.T.n_tiles)

    def step(self, action):
        s,r,d,i = super(CartPoleTileEnv, self).step(action)
        return self._encode(s),r,d,i

    def reset(self):
        s = super(CartPoleTileEnv, self).reset()
        return self._encode(s)

class CartPolePolyEnv(CartPoleEnv):
    """ States are polynomials (1,w,x,y,z,and products) of CartPole 4 inputs."""

    def __init__(self):
        super(CartPolePolyEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = list(itertools.product(range(n_dim), range(n_dim)))
        self.observation_space = spaces.Discrete(1+n_dim+n_dim**2)

    def _converter(self, s):
        """ Converts the 4-element state into the polynomial one."""
        double_products = [s[a]*s[b] for a,b in self.prod_indices]
        return np.array([1] + list(s) + double_products)

    def step(self, action):
        s,r,d,i = super(CartPolePolyEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(CartPolePolyEnv, self).reset()
        return self._converter(s)

class CartPoleDoubleEnv(CartPoleEnv):
    """ Combine both Tiles and Poly for ActorCritic-like methods"""

    def __init__(self):
        super(CartPoleDoubleEnv, self).__init__()
        n_dim = self.observation_space.shape[0]
        self.prod_indices = list(itertools.product(range(n_dim), range(n_dim)))
        n_dim = self.observation_space.shape[0]
        tiles_per_dim = [5]*n_dim
        lims = [[-2,2], [-3,3], [-.5, .5], [-4,4]]
        tilings = n_dim
        self.T = TileCoder(tiles_per_dim, lims, tilings)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1+n_dim+n_dim**2),
            spaces.Discrete(self.T.n_tiles)
        ))

    def _converter(self, s):
        """ Converts the 4-element state into the polynomial one."""
        double_products = [s[a]*s[b] for a,b in self.prod_indices]
        return (np.array([1] + list(s) + double_products), one_hot(self.T[s], self.T.n_tiles))

    def step(self, action):
        s,r,d,i = super(CartPoleDoubleEnv, self).step(action)
        return self._converter(s),r,d,i

    def reset(self):
        s = super(CartPoleDoubleEnv, self).reset()
        return self._converter(s)
