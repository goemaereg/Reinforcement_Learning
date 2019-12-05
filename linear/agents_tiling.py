""" Agents that take into account that the input comes as Sparse Coding
(in particular, tiling) of the state space; hence saving lots of computation.
"""
import numpy as np
from utils import assert_not_abstract, my_argmax, prod
from agents_core import Agent
import random
from collections import deque

class Sarsa(Agent):
    """ Linear method that act eps-greedy and uses TD(0) as update """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(Sarsa, self).__init__(env_shapes) # inits shapes
        self.name = 'Sarsa'
        self.input_dim = prod(self.input_shape) # flattens the shape
        print((self.input_dim, self.n_actions))
        self.explo_horizon = explo_horizon
        self.min_eps = min_eps
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.reset()

    def reset(self):
#        self.w = np.random.randn(self.input_dim, self.n_actions)*0.01 # arb init at 0
        self.w = np.zeros((self.input_dim, self.n_actions)) # arb init at 0
        self.anneal_epsilon(0)
        self.step = 0
        self.verbose = False

    def _q_hat(self, s_idx, a=None):
        """ Evaluates our q function for all actions, taking into account that
        the state comes as an array of indices; the Tile Coding being
        constructed so (one_hot) that we just need to sum w over those indices.
        """
        q_s = np.sum(self.w[s_idx], axis=0)
        return q_s if a is None else q_s[a]

    def act(self, x):
        """ Epsilon-greedy policy over w. In Tile Coding, x is an array of
        indices corresponding to the activated tiles. """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            action_values = self._q_hat(x)
            if self.verbose and np.random.rand()<0.1:
                print("Q values of possible actions: {}".format(action_values))
            return my_argmax(action_values)

    def anneal_epsilon(self, ep:int):
        """ Anneals epsilon linearly based on the step number """
        self.epsilon = max((self.explo_horizon - ep)/self.explo_horizon, self.min_eps)

    def learn(self, x, a, r, x_, d=None):
        """ Updates the weight vector based on the x,a,r,x_ transition,
        where x is the feature vector of currnt state s.
        Updates the annealing epsilon. """
        delta = r + self.gamma*self._q_hat(x_, self.act(x_)) - self._q_hat(x, a)
        self.w[x,a] += self.learn_rate*delta
        # anneal epsilon
        self.step += 1
        self.anneal_epsilon(self.step)

    def tell_specs(self) -> str:
        """ Specifies the specs of the agent (hyperparameters mainly) """
        return "eps={}, learn_rate={}, gamma={}"\
               .format(self.min_eps, self.learn_rate, self.gamma)


class ExpectedSarsa(Sarsa):
    """ Improvement on Sarsa to compute the expectation over all actions """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(ExpectedSarsa, self).__init__(env_shapes, min_eps, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'ExpectedSarsa'

    def learn(self, x, a, r, x_, d=None):
        """ Updates the w based on the x,a,r,x_ transition.
        ExpectedSarsa takes an expectation over possible action probabilities.
        Updates the annealing epsilon. """
        q_hat_x_ = self._q_hat(x_)
        expectation = self.epsilon/self.n_actions*np.sum(q_hat_x_)\
                      + (1-self.epsilon)*np.max(q_hat_x_)
        delta = r + self.gamma*expectation - self._q_hat(x, a)
        self.w[x,a] += self.learn_rate*delta
        # anneal epsilon
        self.step += 1
        self.anneal_epsilon(self.step)

class QLearning(Sarsa):
    """ Improvement on Sarsa to max Q(S',.) over all actions """
    def __init__(self, env_shapes, min_eps=0.1, learn_rate=2.5e-4, gamma=0.9,
                 explo_horizon=50000, **kwargs):
        super(QLearning, self).__init__(env_shapes, min_eps, learn_rate, gamma, explo_horizon) # inits shapes
        self.name = 'QLearning'

    def learn(self, x, a, r, x_, d=None):
        """ Updates the w based on the x,a,r,x_ transition.
        QLearning maxes over actions in the future state (off policy).
        Updates the annealing epsilon. """
        delta = r + self.gamma*max(self._q_hat(x_)) - self._q_hat(x, a)
        self.w[x,a] += self.learn_rate*delta
        self.step += 1
        self.anneal_epsilon(self.step)
