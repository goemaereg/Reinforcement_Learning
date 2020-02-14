import numpy as np
from utils import *
from agents_core import Agent, Random_Agent
from mdp_core import *
from .agents import *
import random

class Intrinsic_Reward_Function(object):
    """ A function that outputs an intrinsic reward given a transition."""

    def __init__(self, env_shapes:tuple, **kwargs):
        super(Intrinsic_Reward_Function, self).__init__()
        self.input_shape, self.n_actions = env_shapes
        self.reset() # inits variables

    def reset(self):
        """ Initializes/resets the variables, if any."""
        pass

    def give_reward(self, s:State, a:Action, s_:State) -> float:
        """ Main function, that outputs the reward.
        It also contains whatever updates on the internal variables that we want. """
        raise NotImplementedError("Accessing give_reward method in abstract class")


class Inverse_sqrt(Intrinsic_Reward_Function):
    """ Visit-count based Intrinsic_Reward_Function, the reward is simply
    r = 1/sqrt(visit counts). """

    def reset(self):
        self.visit_counts = np.zeros(self.input_shape)

    def give_reward(self, s, a, s_):
        """ Updates the visit counts and provides the reward as 1/sqrt(n)"""
        self.visit_counts[s_] += 1
        return 1/np.sqrt(self.visit_counts[s_])


class Inverse_lin(Intrinsic_Reward_Function):
    """ Visit-count based Intrinsic_Reward_Function, the reward is simply
    r = 1/visit counts. """

    def reset(self):
        self.visit_counts = np.zeros(self.input_shape)

    def give_reward(self, s, a, s_):
        """ Updates the visit counts and provides the reward as 1/sqrt(n)"""
        self.visit_counts[s_] += 1
        return 1/self.visit_counts[s_]


class Negative_sqrt(Intrinsic_Reward_Function):
    """ Visit-count based Intrinsic_Reward_Function, the reward is simply
    r = -sqrt(visit counts). """

    def reset(self):
        self.visit_counts = np.zeros(self.input_shape)

    def give_reward(self, s, a, s_):
        """ Updates the visit counts and provides the reward as 1/sqrt(n)"""
        self.visit_counts[s_] += 1
        return -np.sqrt(self.visit_counts[s_])

class Successor_Rep(Intrinsic_Reward_Function):
    """ Reward is 1/(norm of the successor state representation). """

    def __init__(self, env_shapes, gamma=0.9, learn_rate=0.1, **kwargs):
        self.gamma = gamma
        self.learn_rate = learn_rate
        print(kwargs)
        super(Successor_Rep, self).__init__(env_shapes)

    def reset(self):
        self.successor_vs = np.zeros((*self.input_shape, *self.input_shape))

    def give_reward(self, s, a, s_):
        """ Updates the visit counts and provides the reward as 1/sqrt(n)"""
        target = self.gamma*self.successor_vs[s_] # TD(0) bootstrapping
        target[s] += 1 # reward = indicator function that s_t == s
        self.successor_vs[s] += self.learn_rate*(target - self.successor_vs[s])
        return 1/np.sum(np.abs(self.successor_vs[s]))

class Successor_Rep_Neg(Successor_Rep):
    """ Reward is  -(norm of the successor state representation). """

    def reset(self):
        self.successor_vs = np.zeros((*self.input_shape, *self.input_shape))

    def give_reward(self, s, a, s_):
        """ Updates the visit counts and provides the reward as 1/sqrt(n)"""
        target = self.gamma*self.successor_vs[s_] # TD(0) bootstrapping
        target[s] += 1 # reward = indicator function that s_t == s
        self.successor_vs[s] += self.learn_rate*(target - self.successor_vs[s])
        return -np.sum(np.abs(self.successor_vs[s]))
